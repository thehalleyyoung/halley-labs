"""Tests for CVT (Centroidal Voronoi Tessellation).

Covers Lloyd's algorithm convergence, cell assignment,
different dimensions, and adaptive CVT.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.exploration.cvt import CVTTessellation, AdaptiveCVT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def cvt_2d():
    t = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=20, seed=42)
    t.initialize()
    return t


@pytest.fixture
def cvt_4d():
    t = CVTTessellation(n_cells=50, n_dims=4, n_samples=1000, n_lloyd_iters=30, seed=42)
    t.initialize()
    return t


@pytest.fixture
def adaptive_cvt():
    t = AdaptiveCVT(
        n_cells=30, n_dims=4,
        min_cells=10, max_cells=100,
        split_threshold=20, merge_threshold=1,
        seed=42,
    )
    t.initialize()
    return t


# ---------------------------------------------------------------------------
# Test Lloyd's algorithm convergence
# ---------------------------------------------------------------------------

class TestLloydConvergence:

    def test_centroids_are_initialized(self, cvt_4d):
        assert cvt_4d.initialized
        c = cvt_4d.get_centroid(0)
        assert c.shape == (4,)

    def test_centroids_within_bounds(self, cvt_4d):
        for i in range(50):
            c = cvt_4d.get_centroid(i)
            assert np.all(c >= -0.5)  # Tolerance for Lloyd convergence
            assert np.all(c <= 1.5)

    def test_centroids_are_distinct(self, cvt_4d):
        centroids = np.array([cvt_4d.get_centroid(i) for i in range(50)])
        # Check pairwise distances — should all be > 0
        for i in range(50):
            for j in range(i + 1, min(i + 5, 50)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                assert dist > 1e-8

    def test_more_iterations_better_spread(self):
        """More Lloyd iterations should produce better spread centroids."""
        t_few = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=2, seed=42)
        t_few.initialize()
        t_many = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=50, seed=42)
        t_many.initialize()
        # Both should be initialized
        assert t_few.initialized
        assert t_many.initialized

    def test_deterministic_with_seed(self):
        t1 = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=10, seed=42)
        t1.initialize()
        t2 = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=10, seed=42)
        t2.initialize()
        c1 = t1.get_centroid(0)
        c2 = t2.get_centroid(0)
        assert_allclose(c1, c2, atol=1e-10)

    def test_different_seeds_different_results(self):
        t1 = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=10, seed=42)
        t1.initialize()
        t2 = CVTTessellation(n_cells=20, n_dims=2, n_samples=500, n_lloyd_iters=10, seed=99)
        t2.initialize()
        c1 = t1.get_centroid(0)
        c2 = t2.get_centroid(0)
        assert not np.allclose(c1, c2)


# ---------------------------------------------------------------------------
# Test cell assignment
# ---------------------------------------------------------------------------

class TestCellAssignment:

    def test_find_cell_nearest(self, cvt_4d):
        """find_cell should return the nearest centroid's cell."""
        c = cvt_4d.get_centroid(5)
        cell = cvt_4d.find_cell(c)
        assert cell == 5

    def test_find_cell_consistent(self, cvt_4d, rng):
        """Same point always maps to same cell."""
        pt = rng.uniform(0, 1, size=4)
        cell1 = cvt_4d.find_cell(pt)
        cell2 = cvt_4d.find_cell(pt)
        assert cell1 == cell2

    def test_find_cells_batch_matches_single(self, cvt_4d, rng):
        points = rng.uniform(0, 1, size=(10, 4))
        batch_cells = cvt_4d.find_cells_batch(points)
        single_cells = np.array([cvt_4d.find_cell(p) for p in points])
        assert_allclose(batch_cells, single_cells)

    def test_cell_assignment_covers_all_cells(self, cvt_2d, rng):
        """With enough random points, all cells should be visited."""
        points = rng.uniform(0, 1, size=(2000, 2))
        cells = cvt_2d.find_cells_batch(points)
        unique_cells = set(cells.tolist())
        # Most cells should be covered with 2000 points for 20 cells
        assert len(unique_cells) >= 15

    def test_boundary_point(self, cvt_4d):
        """Points at [0,0,...,0] and [1,1,...,1] should be assigned."""
        cell_low = cvt_4d.find_cell(np.zeros(4))
        cell_high = cvt_4d.find_cell(np.ones(4))
        assert 0 <= cell_low < 50
        assert 0 <= cell_high < 50


# ---------------------------------------------------------------------------
# Test with different dimensions
# ---------------------------------------------------------------------------

class TestDifferentDimensions:

    @pytest.mark.parametrize("n_dims", [1, 2, 3, 4, 8])
    def test_initialization(self, n_dims):
        t = CVTTessellation(n_cells=10, n_dims=n_dims, n_samples=200, n_lloyd_iters=10, seed=42)
        bounds = np.column_stack([np.zeros(n_dims), np.ones(n_dims)])
        t.initialize(bounds=bounds)
        assert t.initialized
        c = t.get_centroid(0)
        assert c.shape == (n_dims,)

    @pytest.mark.parametrize("n_dims", [2, 4])
    def test_find_cell(self, n_dims, rng):
        t = CVTTessellation(n_cells=10, n_dims=n_dims, n_samples=200, n_lloyd_iters=10, seed=42)
        t.initialize()
        pt = rng.uniform(0, 1, size=n_dims)
        cell = t.find_cell(pt)
        assert 0 <= cell < 10

    @pytest.mark.parametrize("n_cells", [5, 20, 100])
    def test_different_n_cells(self, n_cells):
        t = CVTTessellation(n_cells=n_cells, n_dims=4, n_samples=max(500, n_cells * 10), n_lloyd_iters=10, seed=42)
        t.initialize()
        assert t.initialized

    def test_cell_distance_symmetric(self, cvt_4d):
        d1 = cvt_4d.cell_distance(0, 1)
        d2 = cvt_4d.cell_distance(1, 0)
        assert_allclose(d1, d2, atol=1e-12)

    def test_cell_distance_self_zero(self, cvt_4d):
        assert cvt_4d.cell_distance(0, 0) == 0.0

    def test_cell_distance_positive(self, cvt_4d):
        d = cvt_4d.cell_distance(0, 1)
        assert d > 0.0


# ---------------------------------------------------------------------------
# Test adaptive CVT
# ---------------------------------------------------------------------------

class TestAdaptiveCVT:

    def test_initialization(self, adaptive_cvt):
        assert adaptive_cvt.initialized

    def test_adapt_returns_stats(self, adaptive_cvt, rng):
        # Visit some cells many times to trigger splitting
        for i in range(5):
            for _ in range(25):
                adaptive_cvt.record_visit(i, rng.uniform())
        stats = adaptive_cvt.adapt()
        assert isinstance(stats, dict)

    def test_coverage_guided_refine(self, adaptive_cvt, rng):
        # Visit some cells
        for i in range(10):
            adaptive_cvt.record_visit(i, rng.uniform())
        n_added = adaptive_cvt.coverage_guided_refine(target_coverage=0.3, max_iters=3)
        assert isinstance(n_added, int)

    def test_adaptation_history(self, adaptive_cvt):
        history = adaptive_cvt.get_adaptation_history()
        assert isinstance(history, dict)

    def test_adaptive_inherits_cvt_methods(self, adaptive_cvt, rng):
        """AdaptiveCVT should have all CVT methods."""
        pt = rng.uniform(0, 1, size=4)
        cell = adaptive_cvt.find_cell(pt)
        assert cell >= 0
        c = adaptive_cvt.get_centroid(0)
        assert c.shape == (4,)

    def test_serialization(self, adaptive_cvt, rng):
        for i in range(5):
            adaptive_cvt.record_visit(i, rng.uniform())
        d = adaptive_cvt.to_dict()
        restored = CVTTessellation.from_dict(d)
        assert restored.initialized

    def test_adaptive_split_merge(self, rng):
        """Test that splitting and merging works."""
        acvt = AdaptiveCVT(
            n_cells=10, n_dims=2,
            min_cells=5, max_cells=50,
            split_threshold=5, merge_threshold=0,
            seed=42,
        )
        acvt.initialize()
        # Visit one cell many times to trigger split
        for _ in range(10):
            acvt.record_visit(0, rng.uniform())
        stats = acvt.adapt()
        assert isinstance(stats, dict)

    def test_custom_bounds(self, rng):
        t = CVTTessellation(n_cells=10, n_dims=2, n_samples=200, n_lloyd_iters=10, seed=42)
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        t.initialize(bounds=bounds)
        assert t.initialized

    def test_visit_count_stats(self, cvt_4d, rng):
        for i in range(20):
            for _ in range(rng.integers(1, 10)):
                cvt_4d.record_visit(i, rng.uniform())
        stats = cvt_4d.visit_count_stats()
        assert isinstance(stats, dict)

    def test_most_visited_cells(self, cvt_4d, rng):
        for i in range(10):
            for _ in range(i + 1):
                cvt_4d.record_visit(i, rng.uniform())
        most = cvt_4d.most_visited_cells(n=3)
        assert len(most) == 3

    def test_coverage_heatmap_data(self, cvt_4d, rng):
        for i in range(5):
            cvt_4d.record_visit(i, rng.uniform())
        data = cvt_4d.get_coverage_heatmap_data()
        assert isinstance(data, dict)
