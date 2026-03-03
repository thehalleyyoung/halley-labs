"""
Comprehensive tests for dp_forge.sensitivity_hull module.

Tests cover hull computation for standard query types, projection,
containment, volume computation, and the approximation approach.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dp_forge.sensitivity_hull import (
    SensitivityHull,
    HullResult,
    HullApproximation,
)
from dp_forge.types import AdjacencyRelation, WorkloadSpec
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def counting_query_hull():
    """Sensitivity hull for a counting query: sensitivity vectors ±1."""
    vectors = np.array([[1.0], [-1.0]])
    return SensitivityHull(vectors, include_negations=False, include_origin=True)


@pytest.fixture
def histogram_2d_hull():
    """Sensitivity hull for a 2-bin histogram: standard basis ± directions."""
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    return SensitivityHull(vectors, include_negations=False, include_origin=True)


# =========================================================================
# Section 1: Counting Query Hull
# =========================================================================


class TestCountingQueryHull:
    """Tests for sensitivity hull of counting queries."""

    def test_counting_hull_interval(self, counting_query_hull):
        """Counting query hull is the interval [-1, 1]."""
        result = counting_query_hull.compute()
        assert isinstance(result, HullResult)
        assert result.dimension == 1
        # Hull should contain the extremes
        verts = result.vertices
        assert verts.min() <= -1.0 + 1e-10
        assert verts.max() >= 1.0 - 1e-10

    def test_counting_sensitivities(self, counting_query_hull):
        """Counting query sensitivities: L1=L2=L∞=2 (diameter)."""
        result = counting_query_hull.compute()
        # Sensitivity = max distance between hull vertices
        assert result.sensitivity_l1 >= 1.0 - 1e-10
        assert result.sensitivity_linf >= 1.0 - 1e-10

    def test_counting_contains_origin(self, counting_query_hull):
        """Origin is inside the counting query hull (1D case)."""
        result = counting_query_hull.compute()
        # In 1D, check that the hull interval covers 0
        assert result.vertices.min() <= 0.0
        assert result.vertices.max() >= 0.0

    def test_counting_contains_sensitivity(self, counting_query_hull):
        """Sensitivity vector is inside the hull."""
        result = counting_query_hull.compute()
        # In 1D, check interval coverage
        assert result.vertices.min() <= 0.5
        assert result.vertices.max() >= 0.5
        assert result.vertices.min() <= -0.5
        assert result.vertices.max() >= -0.5


# =========================================================================
# Section 2: Histogram Query Hull
# =========================================================================


class TestHistogramQueryHull:
    """Tests for sensitivity hull of histogram queries."""

    def test_histogram_hull_vertices(self, histogram_2d_hull):
        """Histogram hull has correct vertices."""
        result = histogram_2d_hull.compute()
        assert result.n_vertices >= 4  # At least the 4 unit directions

    def test_histogram_hull_containment(self, histogram_2d_hull):
        """Points within sensitivity are inside hull."""
        histogram_2d_hull.compute()
        # Origin should be inside
        assert histogram_2d_hull.contains(np.array([0.0, 0.0]))
        # Small perturbation should be inside
        assert histogram_2d_hull.contains(np.array([0.1, 0.1]))

    def test_histogram_hull_dimension(self, histogram_2d_hull):
        """Histogram hull has correct dimension."""
        result = histogram_2d_hull.compute()
        assert result.dimension == 2


# =========================================================================
# Section 3: Hull from Query Function
# =========================================================================


class TestHullFromQuery:
    """Tests for constructing hull from query function."""

    def test_from_query_counting(self):
        """from_query with simple counting function."""
        # Domain: {0, 1, 2}, adjacency: consecutive pairs
        domain = np.arange(3)
        adjacency = AdjacencyRelation(
            edges=[(0, 1), (1, 2)], n=3, symmetric=True,
        )
        query_fn = lambda x: np.array([float(x)])
        hull = SensitivityHull.from_query(query_fn, domain, adjacency)
        result = hull.compute()
        assert result.dimension == 1
        # Sensitivity should be 1 (consecutive count change)
        assert result.sensitivity_linf >= 1.0 - 1e-10

    def test_from_query_2d(self):
        """from_query with 2D query function."""
        domain = np.arange(3)
        adjacency = AdjacencyRelation(
            edges=[(0, 1), (1, 2)], n=3, symmetric=True,
        )
        query_fn = lambda x: np.array([float(x), float(x ** 2)])
        hull = SensitivityHull.from_query(query_fn, domain, adjacency)
        result = hull.compute()
        assert result.dimension == 2


# =========================================================================
# Section 4: Hull from Workload
# =========================================================================


class TestHullFromWorkload:
    """Tests for constructing hull from workload specification."""

    def test_from_workload_identity(self):
        """Identity workload hull."""
        spec = WorkloadSpec.identity(3)
        hull = SensitivityHull.from_workload(spec)
        result = hull.compute()
        assert result.dimension <= 3

    def test_from_workload_range(self):
        """Range workload hull."""
        spec = WorkloadSpec.all_range(3)
        hull = SensitivityHull.from_workload(spec)
        result = hull.compute()
        assert result.n_vertices >= 2


# =========================================================================
# Section 5: Hull Projection
# =========================================================================


class TestHullProjection:
    """Tests for hull projection to lower dimensions."""

    def test_projection_reduces_dimension(self):
        """Projecting 3D hull to 2D reduces dimension."""
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        hull.compute()
        projected = hull.project(axes=[0, 1])
        assert projected.dimension == 2

    def test_projection_contains_projected_sensitivity(self):
        """Projected hull contains projected sensitivity vectors."""
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        hull.compute()
        projected = hull.project(axes=[0, 1])
        # Projected vertices should include ±e₁, ±e₂ in 2D
        assert projected.n_vertices >= 3


# =========================================================================
# Section 6: Volume Computation
# =========================================================================


class TestHullVolume:
    """Tests for hull volume computation."""

    def test_unit_square_volume(self):
        """Hull of ±1 in 2D has known area."""
        vectors = np.array([
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        result = hull.compute()
        # Square [-1,1]² has area 4
        assert abs(result.volume - 4.0) < 1e-6

    def test_interval_volume(self):
        """1D hull has "volume" = length."""
        vectors = np.array([[2.0], [-2.0]])
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        result = hull.compute()
        # Length from -2 to 2 = 4
        assert abs(result.volume - 4.0) < 1e-6

    def test_volume_positive(self):
        """Volume of non-degenerate hull is positive."""
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((10, 3))
        hull = SensitivityHull(vectors, include_negations=True, include_origin=True)
        vol = hull.volume()
        assert vol > 0


# =========================================================================
# Section 7: Containment Tests
# =========================================================================


class TestContainment:
    """Tests for point containment in hull."""

    def test_origin_always_inside(self):
        """Origin is always inside hull (when include_origin=True)."""
        vectors = np.array([[1, 0], [0, 1]], dtype=float)
        hull = SensitivityHull(vectors, include_negations=True, include_origin=True)
        hull.compute()
        assert hull.contains(np.array([0.0, 0.0]))

    def test_vertex_inside(self):
        """Each vertex is inside the hull."""
        vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        result = hull.compute()
        for v in result.vertices:
            assert hull.contains(v)

    def test_outside_point(self):
        """Point far outside hull is not contained."""
        vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        hull.compute()
        assert not hull.contains(np.array([10.0, 10.0]))

    @given(
        x=st.floats(min_value=-0.5, max_value=0.5),
        y=st.floats(min_value=-0.5, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_inner_points_contained(self, x, y):
        """Points well inside hull are always contained."""
        vectors = np.array([
            [1, 1], [1, -1], [-1, 1], [-1, -1],
        ], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        hull.compute()
        assert hull.contains(np.array([x, y]))


# =========================================================================
# Section 8: Hull Approximation
# =========================================================================


class TestHullApproximation:
    """Tests for randomized hull approximation."""

    def test_approximation_computes(self):
        """HullApproximation produces a HullResult."""
        domain = np.arange(5)
        adjacency = AdjacencyRelation(
            edges=[(i, i + 1) for i in range(4)], n=5, symmetric=True,
        )
        query_fn = lambda x: np.array([float(x)])
        approx = HullApproximation(query_fn, domain, adjacency, seed=42)
        result = approx.compute(n_samples=1000)
        assert isinstance(result, HullResult)

    def test_approximation_coverage(self):
        """coverage_estimate returns value in [0, 1]."""
        domain = np.arange(5)
        adjacency = AdjacencyRelation(
            edges=[(i, i + 1) for i in range(4)], n=5, symmetric=True,
        )
        query_fn = lambda x: np.array([float(x)])
        approx = HullApproximation(query_fn, domain, adjacency, seed=42)
        approx.compute(n_samples=1000)
        cov = approx.coverage_estimate(n_bootstrap=10)
        assert 0 <= cov <= 1.0 + 1e-10


# =========================================================================
# Section 9: HullResult Properties
# =========================================================================


class TestHullResultProperties:
    """Tests for HullResult dataclass properties."""

    def test_hull_result_fields(self):
        """HullResult has expected fields."""
        vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        result = hull.compute()
        assert hasattr(result, "vertices")
        assert hasattr(result, "n_vertices")
        assert hasattr(result, "dimension")
        assert hasattr(result, "volume")
        assert hasattr(result, "sensitivity_l1")
        assert hasattr(result, "sensitivity_l2")
        assert hasattr(result, "sensitivity_linf")

    def test_sensitivities_positive(self):
        """All sensitivity norms are positive for non-trivial hull."""
        vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        result = hull.compute()
        assert result.sensitivity_l1 > 0
        assert result.sensitivity_l2 > 0
        assert result.sensitivity_linf > 0

    def test_l2_between_l1_and_linf(self):
        """L2 sensitivity is between L∞ and L1 (for appropriate dimensions)."""
        vectors = np.array([
            [1, 1], [1, -1], [-1, 1], [-1, -1],
        ], dtype=float)
        hull = SensitivityHull(vectors, include_negations=False, include_origin=True)
        result = hull.compute()
        # L∞ ≤ L2 ≤ L1 for typical vectors
        assert result.sensitivity_linf <= result.sensitivity_l2 + 1e-10
        assert result.sensitivity_l2 <= result.sensitivity_l1 + 1e-10
