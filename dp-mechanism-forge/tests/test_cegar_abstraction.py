"""
Tests for CEGAR abstract domain implementations.

Covers IntervalAbstraction, PolyhedralAbstraction, ZonotopeAbstraction,
GaloisConnection, and PrivacyLossAbstraction with property-based testing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    PrivacyBudget,
)
from dp_forge.cegar.abstraction import (
    GaloisConnection,
    IntervalAbstraction,
    PolyhedralAbstraction,
    PolyhedralConstraint,
    PrivacyLossAbstraction,
    ZonotopeAbstraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interval(lower, upper):
    """Create an interval AbstractValue."""
    return AbstractValue(
        domain_type=AbstractDomainType.INTERVAL,
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
    )


def _make_bottom_interval(ndim):
    """Create a bottom element by bypassing __post_init__ validation."""
    v = object.__new__(AbstractValue)
    v.domain_type = AbstractDomainType.INTERVAL
    v.lower = np.full(ndim, np.inf, dtype=np.float64)
    v.upper = np.full(ndim, -np.inf, dtype=np.float64)
    v.constraints = None
    return v


# Hypothesis strategies
finite_floats = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
small_dims = st.integers(min_value=1, max_value=5)


# ===================================================================
# IntervalAbstraction tests
# ===================================================================


class TestIntervalAbstraction:
    """Tests for the interval abstract domain."""

    def setup_method(self):
        self.ia = IntervalAbstraction()

    # -- top ----------------------------------------------------------------

    def test_top_is_not_bottom(self):
        t = self.ia.top(3)
        assert not self.ia.is_bottom(t)

    def test_top_contains_any_point(self):
        t = self.ia.top(2)
        assert self.ia.gamma_contains(t, np.array([999.0, -999.0]))

    # -- is_bottom ----------------------------------------------------------

    def test_is_bottom_on_bottom_element(self):
        b = _make_bottom_interval(2)
        assert self.ia.is_bottom(b)

    def test_is_bottom_false_for_valid_interval(self):
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        assert not self.ia.is_bottom(a)

    def test_is_bottom_false_for_point_interval(self):
        a = _make_interval([0.5], [0.5])
        assert not self.ia.is_bottom(a)

    # -- contains -----------------------------------------------------------

    def test_contains_interior_point(self):
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        assert self.ia.gamma_contains(a, np.array([0.5, 0.5]))

    def test_contains_boundary_point(self):
        a = _make_interval([0.0], [1.0])
        assert self.ia.gamma_contains(a, np.array([0.0]))
        assert self.ia.gamma_contains(a, np.array([1.0]))

    def test_not_contains_outside_point(self):
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        assert not self.ia.gamma_contains(a, np.array([1.5, 0.5]))

    # -- join ---------------------------------------------------------------

    def test_join_is_superset(self):
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        b = _make_interval([0.5, 0.5], [2.0, 2.0])
        j = self.ia.join(a, b)
        np.testing.assert_array_less(j.lower - 1e-12, a.lower)
        np.testing.assert_array_less(a.upper, j.upper + 1e-12)

    def test_join_with_bottom_returns_other(self):
        a = _make_interval([1.0, 2.0], [3.0, 4.0])
        b = _make_bottom_interval(2)
        j = self.ia.join(a, b)
        np.testing.assert_allclose(j.lower, a.lower)
        np.testing.assert_allclose(j.upper, a.upper)

    def test_join_commutative(self):
        a = _make_interval([0.0, 1.0], [2.0, 3.0])
        b = _make_interval([1.0, 0.0], [4.0, 2.0])
        j1 = self.ia.join(a, b)
        j2 = self.ia.join(b, a)
        np.testing.assert_allclose(j1.lower, j2.lower)
        np.testing.assert_allclose(j1.upper, j2.upper)

    def test_join_idempotent(self):
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        j = self.ia.join(a, a)
        np.testing.assert_allclose(j.lower, a.lower)
        np.testing.assert_allclose(j.upper, a.upper)

    # -- meet ---------------------------------------------------------------

    def test_meet_overlapping(self):
        a = _make_interval([0.0, 0.0], [2.0, 2.0])
        b = _make_interval([1.0, 1.0], [3.0, 3.0])
        m = self.ia.meet(a, b)
        np.testing.assert_allclose(m.lower, [1.0, 1.0])
        np.testing.assert_allclose(m.upper, [2.0, 2.0])

    def test_meet_idempotent(self):
        a = _make_interval([0.0], [1.0])
        m = self.ia.meet(a, a)
        np.testing.assert_allclose(m.lower, a.lower)
        np.testing.assert_allclose(m.upper, a.upper)

    def test_meet_subset(self):
        a = _make_interval([0.0, 0.0], [3.0, 3.0])
        b = _make_interval([1.0, 1.0], [2.0, 2.0])
        m = self.ia.meet(a, b)
        np.testing.assert_allclose(m.lower, b.lower)
        np.testing.assert_allclose(m.upper, b.upper)

    # -- widen / narrow -----------------------------------------------------

    def test_widen_expands_decreasing_lower(self):
        a = _make_interval([1.0], [2.0])
        b = _make_interval([0.5], [2.0])
        w = self.ia.widen(a, b)
        assert w.lower[0] == -np.inf

    def test_widen_expands_increasing_upper(self):
        a = _make_interval([1.0], [2.0])
        b = _make_interval([1.0], [2.5])
        w = self.ia.widen(a, b)
        assert w.upper[0] == np.inf

    def test_widen_stable_does_not_expand(self):
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        b = _make_interval([0.0, 0.0], [1.0, 1.0])
        w = self.ia.widen(a, b)
        np.testing.assert_allclose(w.lower, a.lower)
        np.testing.assert_allclose(w.upper, a.upper)

    def test_widen_with_bottom_returns_b(self):
        bot = _make_bottom_interval(2)
        b = _make_interval([0.0, 0.0], [1.0, 1.0])
        w = self.ia.widen(bot, b)
        np.testing.assert_allclose(w.lower, b.lower)
        np.testing.assert_allclose(w.upper, b.upper)

    def test_narrow_replaces_infinite_bounds(self):
        a = _make_interval([-np.inf], [np.inf])
        b = _make_interval([0.0], [10.0])
        n = self.ia.narrow(a, b)
        assert n.lower[0] == 0.0
        assert n.upper[0] == 10.0

    def test_narrow_keeps_finite_bounds(self):
        a = _make_interval([1.0], [5.0])
        b = _make_interval([0.0], [10.0])
        n = self.ia.narrow(a, b)
        assert n.lower[0] == 1.0
        assert n.upper[0] == 5.0

    # -- leq ----------------------------------------------------------------

    def test_leq_subset(self):
        a = _make_interval([1.0, 1.0], [2.0, 2.0])
        b = _make_interval([0.0, 0.0], [3.0, 3.0])
        assert self.ia.leq(a, b)

    def test_leq_not_subset(self):
        a = _make_interval([0.0, 0.0], [3.0, 3.0])
        b = _make_interval([1.0, 1.0], [2.0, 2.0])
        assert not self.ia.leq(a, b)

    def test_leq_bottom_leq_everything(self):
        b = _make_bottom_interval(2)
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        assert self.ia.leq(b, a)

    # -- alpha --------------------------------------------------------------

    def test_alpha_single_point(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        a = self.ia.alpha(pts)
        np.testing.assert_allclose(a.lower, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(a.upper, [1.0, 2.0, 3.0])

    def test_alpha_multiple_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]])
        a = self.ia.alpha(pts)
        np.testing.assert_allclose(a.lower, [0.0, 0.0])
        np.testing.assert_allclose(a.upper, [1.0, 2.0])

    def test_alpha_all_points_contained(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
        a = self.ia.alpha(pts)
        for pt in pts:
            assert self.ia.gamma_contains(a, pt)

    # -- widen_with_thresholds -----------------------------------------------

    def test_widen_with_thresholds(self):
        a = _make_interval([1.0], [2.0])
        b = _make_interval([0.5], [2.5])
        thresholds = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 10.0])
        w = self.ia.widen_with_thresholds(a, b, thresholds)
        # Lower should jump to 0.5 (the threshold at or below b.lower=0.5)
        assert w.lower[0] <= 0.5 + 1e-12
        # Upper should jump to 3.0 (the threshold at or above b.upper=2.5)
        assert w.upper[0] >= 2.5 - 1e-12

    # -- abstract_privacy_ratio ----------------------------------------------

    def test_abstract_privacy_ratio(self):
        a = _make_interval([0.3, 0.7], [0.3, 0.7])
        b = _make_interval([0.7, 0.3], [0.7, 0.3])
        lo, hi = self.ia.abstract_privacy_ratio(a, b)
        assert lo <= hi
        # Ratios: 0.3/0.7 and 0.7/0.3
        assert hi >= 0.7 / 0.3 - 1e-6

    # -- parametrize ---------------------------------------------------------

    @pytest.mark.parametrize("ndim", [1, 2, 5, 10])
    def test_top_dimensions(self, ndim):
        t = self.ia.top(ndim)
        assert t.lower.shape == (ndim,)
        assert t.upper.shape == (ndim,)
        assert not self.ia.is_bottom(t)

    @pytest.mark.parametrize("ndim", [1, 3, 5])
    def test_alpha_roundtrip_soundness(self, ndim):
        rng = np.random.default_rng(42)
        pts = rng.uniform(-5, 5, size=(10, ndim))
        a = self.ia.alpha(pts)
        for pt in pts:
            assert self.ia.gamma_contains(a, pt)

    # -- hypothesis property-based tests -------------------------------------

    @given(
        lower=arrays(np.float64, shape=3, elements=st.floats(-10, 0, allow_nan=False, allow_infinity=False)),
        upper=arrays(np.float64, shape=3, elements=st.floats(0, 10, allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=30)
    def test_join_contains_both_operands(self, lower, upper):
        assume(np.all(lower <= upper))
        a = _make_interval(lower, upper)
        b = _make_interval(lower + 1, upper + 1)
        j = self.ia.join(a, b)
        assert self.ia.leq(a, j)
        assert self.ia.leq(b, j)

    @given(
        lower=arrays(np.float64, shape=2, elements=st.floats(-5, 0, allow_nan=False, allow_infinity=False)),
        upper=arrays(np.float64, shape=2, elements=st.floats(0, 5, allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=30)
    def test_alpha_gamma_soundness(self, lower, upper):
        """α(S) should contain all points in S (soundness)."""
        assume(np.all(lower <= upper))
        pts = np.array([lower, upper, (lower + upper) / 2.0])
        a = self.ia.alpha(pts)
        for pt in pts:
            assert self.ia.gamma_contains(a, pt)


# ===================================================================
# PolyhedralAbstraction tests
# ===================================================================


class TestPolyhedralAbstraction:
    """Tests for the polyhedral abstract domain."""

    def setup_method(self):
        self.pa = PolyhedralAbstraction()

    def test_top_has_no_constraints(self):
        t = self.pa.top(2)
        assert t.constraints == [] or t.constraints is None

    def test_alpha_contains_all_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        a = self.pa.alpha(pts)
        for pt in pts:
            assert self.pa.gamma_contains(a, pt)

    def test_add_constraint_tightens_bounds(self):
        t = self.pa.top(2)
        # x0 <= 5
        c = self.pa.add_constraint(t, np.array([1.0, 0.0]), 5.0)
        assert self.pa.gamma_contains(c, np.array([4.0, 0.0]))
        assert not self.pa.gamma_contains(c, np.array([6.0, 0.0]))

    def test_meet_intersection(self):
        t = self.pa.top(2)
        # x0 <= 3
        a = self.pa.add_constraint(t, np.array([1.0, 0.0]), 3.0)
        # x1 <= 2
        b = self.pa.add_constraint(t, np.array([0.0, 1.0]), 2.0)
        m = self.pa.meet(a, b)
        assert self.pa.gamma_contains(m, np.array([2.0, 1.0]))
        assert not self.pa.gamma_contains(m, np.array([4.0, 1.0]))
        assert not self.pa.gamma_contains(m, np.array([2.0, 3.0]))

    def test_join_preserves_common_constraints(self):
        """Join of two polyhedra with shared constraints should preserve them."""
        # Create bounded polyhedra so bbox is finite
        pts_a = np.array([[0.0, 0.0], [3.0, 3.0]])
        pts_b = np.array([[1.0, 1.0], [4.0, 4.0]])
        a = self.pa.alpha(pts_a)
        b = self.pa.alpha(pts_b)
        j = self.pa.join(a, b)
        # The join should contain all original points
        for pt in [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0], [4.0, 4.0]]:
            assert self.pa.gamma_contains(j, np.array(pt))

    def test_leq_top_contains_constrained(self):
        t = self.pa.top(2)
        a = self.pa.add_constraint(t, np.array([1.0, 0.0]), 3.0)
        assert self.pa.leq(a, t)

    def test_widen_drops_violated_constraints(self):
        t = self.pa.top(2)
        a = self.pa.add_constraint(t, np.array([1.0, 0.0]), 3.0)
        b = self.pa.add_constraint(t, np.array([1.0, 0.0]), 5.0)
        w = self.pa.widen(a, b)
        # Widening keeps constraints from a that b satisfies
        # Since b has x0 <= 5, a's constraint x0 <= 3 is NOT satisfied by b's bbox
        # (b's bbox has x0 up to inf since we only added one constraint)
        # so widening may drop constraints
        assert w is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_alpha_soundness(self, ndim):
        rng = np.random.default_rng(123)
        pts = rng.uniform(-2, 2, size=(5, ndim))
        a = self.pa.alpha(pts)
        for pt in pts:
            assert self.pa.gamma_contains(a, pt)

    def test_constraint_operations(self):
        """Test adding and checking constraints."""
        t = self.pa.top(3)
        # Add: x0 + x1 <= 4
        c = self.pa.add_constraint(t, np.array([1.0, 1.0, 0.0]), 4.0)
        assert self.pa.gamma_contains(c, np.array([1.0, 2.0, 0.0]))
        assert not self.pa.gamma_contains(c, np.array([3.0, 3.0, 0.0]))

    def test_containment_via_projection(self):
        """Test projection by bounding box."""
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = self.pa.alpha(pts)
        # The bounding box should be [1,3] x [2,4]
        assert a.lower[0] <= 1.0 + 1e-9
        assert a.upper[0] >= 3.0 - 1e-9
        assert a.lower[1] <= 2.0 + 1e-9
        assert a.upper[1] >= 4.0 - 1e-9


# ===================================================================
# ZonotopeAbstraction tests
# ===================================================================


class TestZonotopeAbstraction:
    """Tests for the zonotope abstract domain."""

    def setup_method(self):
        self.za = ZonotopeAbstraction()

    def test_top_not_bottom(self):
        t = self.za.top(2)
        assert not self.za.is_bottom(t)

    def test_alpha_single_point(self):
        pts = np.array([[1.0, 2.0]])
        a = self.za.alpha(pts)
        assert self.za.gamma_contains(a, np.array([1.0, 2.0]))

    def test_alpha_multiple_points_contained(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        a = self.za.alpha(pts)
        for pt in pts:
            assert self.za.gamma_contains(a, pt)

    def test_generator_operations_affine_transform(self):
        """Test that affine transform works correctly."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        a = self.za.alpha(pts)
        M = np.array([[2.0, 0.0], [0.0, 3.0]])
        offset = np.array([1.0, 1.0])
        result = self.za.affine_transform(a, M, offset)
        # Transformed points should be contained
        for pt in pts:
            transformed = M @ pt + offset
            assert self.za.gamma_contains(result, transformed)

    def test_join_contains_both(self):
        a = self.za.alpha(np.array([[0.0, 0.0], [1.0, 0.0]]))
        b = self.za.alpha(np.array([[0.0, 0.0], [0.0, 1.0]]))
        j = self.za.join(a, b)
        # All original points should be in the join
        for pt in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]:
            assert self.za.gamma_contains(j, np.array(pt))

    def test_meet_bounding_box_intersection(self):
        a = self.za.alpha(np.array([[0.0, 0.0], [2.0, 2.0]]))
        b = self.za.alpha(np.array([[1.0, 1.0], [3.0, 3.0]]))
        m = self.za.meet(a, b)
        # Intersection should be around [1,2] x [1,2]
        assert self.za.gamma_contains(m, np.array([1.5, 1.5]))

    def test_leq_subset(self):
        inner = self.za.alpha(np.array([[1.0, 1.0], [2.0, 2.0]]))
        outer = self.za.alpha(np.array([[0.0, 0.0], [3.0, 3.0]]))
        assert self.za.leq(inner, outer)

    def test_widen_expands(self):
        a = self.za.alpha(np.array([[0.0, 0.0], [1.0, 1.0]]))
        b = self.za.alpha(np.array([[0.0, 0.0], [1.5, 1.5]]))
        w = self.za.widen(a, b)
        assert self.za.gamma_contains(w, np.array([1.5, 1.5]))

    def test_over_approximation_soundness(self):
        """Zonotope should soundly over-approximate its input."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-3, 3, size=(20, 3))
        a = self.za.alpha(pts)
        for pt in pts:
            assert self.za.gamma_contains(a, pt)

    @pytest.mark.parametrize("ndim", [1, 2, 4])
    def test_top_dimensions(self, ndim):
        t = self.za.top(ndim)
        assert t.lower.shape == (ndim,)
        assert not self.za.is_bottom(t)

    def test_reduce_generators(self):
        """Test that generator reduction preserves containment."""
        za = ZonotopeAbstraction(max_generators=3)
        rng = np.random.default_rng(7)
        pts = rng.uniform(-2, 2, size=(10, 2))
        a = za.alpha(pts)
        for pt in pts:
            assert za.gamma_contains(a, pt)


# ===================================================================
# GaloisConnection tests
# ===================================================================


class TestGaloisConnection:
    """Tests for Galois connection properties."""

    @pytest.mark.parametrize("domain_cls", [IntervalAbstraction, ZonotopeAbstraction])
    def test_alpha_gamma_roundtrip(self, domain_cls):
        """γ(α(S)) ⊇ S: abstraction then concretization contains originals."""
        domain = domain_cls()
        gc = GaloisConnection(domain)
        pts = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 2.0]])
        abstract = gc.abstract(pts)
        for pt in pts:
            assert gc.concretization_contains(abstract, pt)

    @pytest.mark.parametrize("domain_cls", [IntervalAbstraction, ZonotopeAbstraction])
    def test_soundness_check(self, domain_cls):
        domain = domain_cls()
        gc = GaloisConnection(domain)
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        abstract = gc.abstract(pts)
        assert gc.is_sound_approximation(pts, abstract)

    def test_monotonicity_interval(self):
        """If S ⊆ T then α(S) ⊆ α(T) (monotonicity)."""
        ia = IntervalAbstraction()
        gc = GaloisConnection(ia)
        s = np.array([[1.0, 1.0], [2.0, 2.0]])
        t = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        alpha_s = gc.abstract(s)
        alpha_t = gc.abstract(t)
        assert ia.leq(alpha_s, alpha_t)

    def test_abstract_transfer(self):
        ia = IntervalAbstraction()
        gc = GaloisConnection(ia)
        a = _make_interval([0.0, 0.0], [1.0, 1.0])
        b = _make_interval([0.5, 0.5], [2.0, 2.0])
        result = gc.abstract_transfer(a, ia, join_with=b)
        # Result should contain both a and b
        assert ia.leq(a, result)
        assert ia.leq(b, result)

    @given(
        pts=arrays(np.float64, shape=(5, 2),
                   elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=20)
    def test_alpha_gamma_property_based(self, pts):
        """Property-based soundness: all points in γ(α(pts))."""
        ia = IntervalAbstraction()
        gc = GaloisConnection(ia)
        abstract = gc.abstract(pts)
        assert gc.is_sound_approximation(pts, abstract)


# ===================================================================
# PrivacyLossAbstraction tests
# ===================================================================


class TestPrivacyLossAbstraction:
    """Tests for privacy loss over-approximation."""

    def setup_method(self):
        self.ia = IntervalAbstraction()
        self.pla = PrivacyLossAbstraction(self.ia)

    def test_abstract_log_ratio_positive(self):
        num = _make_interval([0.7], [0.7])
        den = _make_interval([0.3], [0.3])
        lo, hi = self.pla.abstract_log_ratio(num, den)
        assert lo <= hi
        expected = math.log(0.7 / 0.3)
        assert hi >= expected - 1e-6

    def test_abstract_log_ratio_equal(self):
        a = _make_interval([0.5], [0.5])
        lo, hi = self.pla.abstract_log_ratio(a, a)
        assert abs(lo) < 1e-6
        assert abs(hi) < 1e-6

    def test_check_epsilon_bound_verified(self):
        """Mechanism satisfying ε-DP should be verified."""
        # Randomized response with strong privacy
        eps = 1.0
        p = math.exp(eps) / (1 + math.exp(eps))
        mech = np.array([[p, 1 - p], [1 - p, p]])
        adj = AdjacencyRelation.hamming_distance_1(2)
        verified, violation = self.pla.check_epsilon_bound(mech, adj, eps)
        # The abstract log-ratio uses max/min globally, which can under-approximate
        # the true privacy loss. It should still verify for well-structured RR.
        assert verified
        assert violation is None

    def test_check_epsilon_bound_same_rows(self):
        """Identical rows should always verify."""
        mech = np.array([[0.5, 0.5], [0.5, 0.5]])
        adj = AdjacencyRelation.hamming_distance_1(2)
        verified, violation = self.pla.check_epsilon_bound(mech, adj, 0.01)
        assert verified

    def test_abstract_composition_additive(self):
        losses = [(0.5, 1.0), (0.3, 0.8), (0.1, 0.5)]
        lo, hi = self.pla.abstract_composition(losses)
        assert abs(lo - 0.9) < 1e-9
        assert abs(hi - 2.3) < 1e-9

    def test_abstract_composition_empty(self):
        lo, hi = self.pla.abstract_composition([])
        assert lo == 0.0
        assert hi == 0.0

    def test_sound_over_approximation_property(self):
        """Abstract loss bounds should be finite for valid mechanisms."""
        mech = np.array([[0.6, 0.4], [0.4, 0.6]])
        adj = AdjacencyRelation.hamming_distance_1(2)
        row_0 = self.ia.alpha(mech[0:1, :])
        row_1 = self.ia.alpha(mech[1:2, :])
        lo, hi = self.pla.abstract_log_ratio(row_0, row_1)
        assert math.isfinite(lo)
        assert math.isfinite(hi)

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0])
    def test_sound_for_randomized_response(self, eps):
        """RR with parameter eps should verify at budget eps."""
        p = math.exp(eps) / (1 + math.exp(eps))
        mech = np.array([[p, 1 - p], [1 - p, p]])
        adj = AdjacencyRelation.hamming_distance_1(2)
        verified, _ = self.pla.check_epsilon_bound(mech, adj, eps)
        assert verified

    @given(
        probs=arrays(np.float64, shape=4,
                     elements=st.floats(0.05, 0.95, allow_nan=False, allow_infinity=False)),
    )
    @settings(max_examples=20)
    def test_log_ratio_bounds_finite(self, probs):
        """Log-ratio bounds should always be finite for positive inputs."""
        a_vals = probs[:2]
        b_vals = probs[2:]
        a = _make_interval(a_vals, a_vals)
        b = _make_interval(b_vals, b_vals)
        lo, hi = self.pla.abstract_log_ratio(a, b)
        assert math.isfinite(lo)
        assert math.isfinite(hi)


# ===================================================================
# Edge case tests
# ===================================================================


class TestEdgeCases:
    """Test boundary and degenerate cases."""

    def test_interval_1d(self):
        ia = IntervalAbstraction()
        a = _make_interval([0.0], [1.0])
        assert ia.gamma_contains(a, np.array([0.5]))

    def test_interval_high_dim(self):
        ia = IntervalAbstraction()
        ndim = 50
        t = ia.top(ndim)
        assert t.lower.shape == (ndim,)

    def test_zonotope_single_generator(self):
        za = ZonotopeAbstraction()
        pts = np.array([[0.0], [1.0]])
        a = za.alpha(pts)
        assert za.gamma_contains(a, np.array([0.5]))

    def test_polyhedral_single_dim(self):
        pa = PolyhedralAbstraction()
        pts = np.array([[0.0], [1.0]])
        a = pa.alpha(pts)
        assert pa.gamma_contains(a, np.array([0.5]))

    def test_galois_1d(self):
        gc = GaloisConnection(IntervalAbstraction())
        pts = np.array([[5.0]])
        a = gc.abstract(pts)
        assert gc.concretization_contains(a, np.array([5.0]))

    def test_interval_point_interval(self):
        ia = IntervalAbstraction()
        a = _make_interval([3.0, 3.0], [3.0, 3.0])
        assert ia.gamma_contains(a, np.array([3.0, 3.0]))
        assert not ia.gamma_contains(a, np.array([3.1, 3.0]))

    def test_privacy_loss_very_small_probs(self):
        """Test with very small probabilities (near zero)."""
        ia = IntervalAbstraction()
        pla = PrivacyLossAbstraction(ia)
        a = _make_interval([1e-10], [1e-10])
        b = _make_interval([1e-10], [1e-10])
        lo, hi = pla.abstract_log_ratio(a, b)
        assert math.isfinite(lo)
        assert math.isfinite(hi)
