"""Tests for zonotope abstract domain."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.abstract.zonotope import Zonotope


class TestZonotopeCreation:
    """Test zonotope construction and properties."""

    def test_basic_creation(self):
        """Test creating a zonotope."""
        center = np.array([1.0, 2.0])
        generators = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = Zonotope(center=center, generators=generators)
        assert z.dimension == 2
        assert z.num_generators == 2

    def test_point_zonotope(self):
        """Test zonotope with no generators (a point)."""
        center = np.array([3.0, 4.0])
        generators = np.zeros((0, 2))
        z = Zonotope(center=center, generators=generators)
        assert z.dimension == 2
        assert z.num_generators == 0

    def test_1d_zonotope(self):
        """Test 1D zonotope (an interval)."""
        z = Zonotope(center=np.array([5.0]), generators=np.array([[2.0]]))
        bbox = z.bounding_box()
        assert np.isclose(bbox[0, 0], 3.0)
        assert np.isclose(bbox[0, 1], 7.0)

    def test_bounding_box(self):
        """Test bounding box computation."""
        center = np.array([0.0, 0.0])
        generators = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        z = Zonotope(center=center, generators=generators)
        bbox = z.bounding_box()
        assert bbox.shape == (2, 2)
        assert bbox[0, 0] <= -1.0
        assert bbox[0, 1] >= 1.0

    def test_dimension_consistency(self):
        """Test that center and generators have consistent dimensions."""
        center = np.array([1.0, 2.0, 3.0])
        generators = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        z = Zonotope(center=center, generators=generators)
        assert z.dimension == 3
        assert z.num_generators == 2


class TestZonotopeOperations:
    """Test zonotope arithmetic and geometric operations."""

    def test_affine_transform(self):
        """Test affine transformation z' = Wz + b."""
        z = Zonotope(
            center=np.array([1.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, -1.0])
        z2 = z.affine_transform(W, b)
        assert z2.dimension == 2
        np.testing.assert_allclose(z2.center, [3.0, -1.0])

    def test_affine_transform_dimension_change(self):
        """Test affine transform that changes dimension."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([0.0, 0.0, 0.0])
        z2 = z.affine_transform(W, b)
        assert z2.dimension == 3
        np.testing.assert_allclose(z2.center, [1.0, 2.0, 3.0])

    def test_join(self):
        """Test join (over-approximate convex hull)."""
        z1 = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0]])
        )
        z2 = Zonotope(
            center=np.array([2.0, 0.0]),
            generators=np.array([[1.0, 0.0]])
        )
        z_join = z1.join(z2)
        assert z_join.dimension == 2
        # Should contain points from both
        bbox = z_join.bounding_box()
        assert bbox[0, 0] <= -1.0
        assert bbox[0, 1] >= 3.0

    def test_join_same(self):
        """Test join of identical zonotopes."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        z_join = z.join(z)
        np.testing.assert_allclose(z_join.center, z.center)

    def test_widening(self):
        """Test widening operator."""
        z1 = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        z2 = Zonotope(
            center=np.array([0.1, 0.1]),
            generators=np.array([[1.2, 0.0], [0.0, 1.3]])
        )
        z_wide = z1.widening(z2, threshold=0.5)
        assert z_wide.dimension == 2
        # Widened zonotope should be at least as large as z2
        bbox1 = z2.bounding_box()
        bbox_w = z_wide.bounding_box()
        assert bbox_w[0, 0] <= bbox1[0, 0] + 0.1
        assert bbox_w[0, 1] >= bbox1[0, 1] - 0.1

    def test_minkowski_sum(self):
        """Test Minkowski sum."""
        z1 = Zonotope(
            center=np.array([1.0, 0.0]),
            generators=np.array([[1.0, 0.0]])
        )
        z2 = Zonotope(
            center=np.array([0.0, 1.0]),
            generators=np.array([[0.0, 1.0]])
        )
        z_sum = z1.minkowski_sum(z2)
        np.testing.assert_allclose(z_sum.center, [1.0, 1.0])
        assert z_sum.num_generators == 2

    def test_scale(self):
        """Test zonotope scaling."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        z2 = z.scale(2.0)
        np.testing.assert_allclose(z2.center, [2.0, 4.0])
        np.testing.assert_allclose(z2.generators, [[2.0, 0.0], [0.0, 2.0]])

    def test_project(self):
        """Test projection onto subset of dimensions."""
        z = Zonotope(
            center=np.array([1.0, 2.0, 3.0]),
            generators=np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])
        )
        z_proj = z.project([0, 2])
        assert z_proj.dimension == 2
        np.testing.assert_allclose(z_proj.center, [1.0, 3.0])

    def test_sample(self):
        """Test sampling from zonotope."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        samples = z.sample(100)
        assert samples.shape == (100, 2)
        bbox = z.bounding_box()
        for s in samples:
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= s[d] <= bbox[d, 1] + 0.01


class TestZonotopeContainment:
    """Test point containment and set operations."""

    def test_contains_center(self):
        """Test that center is contained."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        assert z.contains_point(np.array([1.0, 2.0]))

    def test_contains_vertex(self):
        """Test that vertices are contained."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        assert z.contains_point(np.array([1.0, 1.0]))
        assert z.contains_point(np.array([-1.0, -1.0]))

    def test_not_contains_outside(self):
        """Test that points outside are not contained."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        assert not z.contains_point(np.array([2.0, 2.0]))

    def test_meet_halfspace(self):
        """Test intersection with halfspace."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[2.0, 0.0], [0.0, 2.0]])
        )
        a = np.array([1.0, 0.0])
        b = 1.0
        z_meet = z.meet_halfspace(a, b)
        assert z_meet is not None
        bbox = z_meet.bounding_box()
        assert bbox[0, 1] <= 1.0 + 0.5  # Allow some overapproximation


class TestZonotopeGeneratorManagement:
    """Test generator reduction and management."""

    def test_reduce_generators(self):
        """Test Girard's generator reduction."""
        center = np.array([0.0, 0.0])
        gens = np.random.randn(20, 2) * 0.5
        z = Zonotope(center=center, generators=gens)
        z_red = z.reduce_generators(max_gens=4)
        assert z_red.num_generators <= 4
        # Reduced should still contain center
        assert z_red.contains_point(center)

    def test_remove_zero_generators(self):
        """Test removing near-zero generators."""
        center = np.array([0.0, 0.0])
        generators = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1e-15, 1e-15],
        ])
        z = Zonotope(center=center, generators=generators)
        z_clean = z.remove_zero_generators(tol=1e-10)
        assert z_clean.num_generators == 2

    def test_reduce_preserves_containment(self):
        """Test that reduction preserves over-approximation."""
        np.random.seed(42)
        center = np.array([1.0, 2.0, 3.0])
        gens = np.random.randn(15, 3) * 0.3
        z = Zonotope(center=center, generators=gens)
        z_red = z.reduce_generators(max_gens=6)
        # Reduced bounding box should contain original bounding box
        bbox_orig = z.bounding_box()
        bbox_red = z_red.bounding_box()
        for d in range(3):
            assert bbox_red[d, 0] <= bbox_orig[d, 0] + 0.01
            assert bbox_red[d, 1] >= bbox_orig[d, 1] - 0.01


class TestZonotopeSerialization:
    """Test zonotope serialization."""

    def test_to_from_dict(self):
        """Test serialization round-trip."""
        z = Zonotope(
            center=np.array([1.0, 2.0, 3.0]),
            generators=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        d = z.to_dict()
        z2 = Zonotope.from_dict(d)
        np.testing.assert_allclose(z2.center, z.center)
        np.testing.assert_allclose(z2.generators, z.generators)

    def test_2d_vertices(self):
        """Test 2D vertex computation."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        verts = z.vertices_2d([0, 1])
        assert len(verts) >= 4  # Square has 4 vertices


class TestZonotopeHighDimensional:
    """Test zonotope in higher dimensions."""

    def test_high_dim(self):
        """Test zonotope in 10D."""
        d = 10
        center = np.zeros(d)
        generators = np.eye(d) * 0.5
        z = Zonotope(center=center, generators=generators)
        assert z.dimension == d
        bbox = z.bounding_box()
        for i in range(d):
            np.testing.assert_allclose(bbox[i], [-0.5, 0.5])

    def test_many_generators_high_dim(self):
        """Test with many generators in high dim."""
        np.random.seed(123)
        d = 20
        n_gens = 50
        z = Zonotope(
            center=np.zeros(d),
            generators=np.random.randn(n_gens, d) * 0.1
        )
        assert z.dimension == d
        assert z.num_generators == n_gens
        bbox = z.bounding_box()
        assert bbox.shape == (d, 2)
        # All intervals should be finite
        assert np.all(np.isfinite(bbox))
