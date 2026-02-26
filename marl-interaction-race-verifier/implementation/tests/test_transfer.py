"""Tests for abstract transfer functions."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.abstract.zonotope import Zonotope
from marace.abstract.transfer import (
    LinearTransfer,
    ReLUTransfer,
    TanhTransfer,
)


class TestLinearTransfer:
    """Test linear/affine abstract transfer function."""

    def test_identity_transform(self):
        """Test identity transformation."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        lt = LinearTransfer(
            weight=np.eye(2),
            bias=np.zeros(2)
        )
        z2 = lt.apply(z)
        np.testing.assert_allclose(z2.center, z.center)
        np.testing.assert_allclose(z2.generators, z.generators)

    def test_scaling_transform(self):
        """Test scaling transformation."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        lt = LinearTransfer(
            weight=np.array([[2.0, 0.0], [0.0, 3.0]]),
            bias=np.zeros(2)
        )
        z2 = lt.apply(z)
        np.testing.assert_allclose(z2.center, [2.0, 6.0])

    def test_translation_transform(self):
        """Test translation via bias."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0]])
        )
        lt = LinearTransfer(
            weight=np.eye(2),
            bias=np.array([5.0, 3.0])
        )
        z2 = lt.apply(z)
        np.testing.assert_allclose(z2.center, [5.0, 3.0])

    def test_dimension_change(self):
        """Test linear transform changing dimension."""
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        lt = LinearTransfer(
            weight=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            bias=np.zeros(3)
        )
        z2 = lt.apply(z)
        assert z2.dimension == 3
        np.testing.assert_allclose(z2.center, [1.0, 2.0, 3.0])

    def test_soundness_linear(self):
        """Test soundness: all concrete outputs are in abstract output."""
        np.random.seed(42)
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5], [0.3, 0.3]])
        )
        W = np.array([[1.5, -0.5], [0.3, 2.0]])
        b = np.array([0.1, -0.2])
        lt = LinearTransfer(weight=W, bias=b)
        z2 = lt.apply(z)
        for _ in range(100):
            x = z.sample(1)[0]
            y = W @ x + b
            bbox = z2.bounding_box()
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= y[d] <= bbox[d, 1] + 0.01


class TestReLUTransfer:
    """Test ReLU abstract transfer function."""

    def test_positive_region(self):
        """Test ReLU when zonotope is entirely positive."""
        z = Zonotope(
            center=np.array([5.0, 5.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        relu = ReLUTransfer()
        z2 = relu.apply(z)
        # Should be unchanged since all values are positive
        np.testing.assert_allclose(z2.center, z.center, atol=0.1)

    def test_negative_region(self):
        """Test ReLU when zonotope is entirely negative."""
        z = Zonotope(
            center=np.array([-5.0, -5.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        relu = ReLUTransfer()
        z2 = relu.apply(z)
        # Should be near zero
        bbox = z2.bounding_box()
        for d in range(2):
            assert bbox[d, 0] >= -0.01
            assert bbox[d, 1] <= 0.5  # Some overapproximation allowed

    def test_crossing_region(self):
        """Test ReLU when zonotope crosses zero.

        DeepZ parallelogram approximation with l=-2, u=2 gives
        λ=0.5, μ=0.5, producing a sound over-approximation with
        lower bound -1.0 (not 0).
        """
        z = Zonotope(
            center=np.array([0.0]),
            generators=np.array([[2.0]])
        )
        relu = ReLUTransfer()
        z2 = relu.apply(z)
        bbox = z2.bounding_box()
        # DeepZ over-approximation: lower bound is -l*u/(2*(u-l)) = -1.0
        assert bbox[0, 0] >= -1.01
        assert bbox[0, 1] >= 2.0 - 0.01

    def test_soundness_relu(self):
        """Test soundness of ReLU abstract transformer."""
        np.random.seed(42)
        z = Zonotope(
            center=np.array([0.5, -0.5]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0], [0.3, 0.3]])
        )
        relu = ReLUTransfer()
        z2 = relu.apply(z)
        bbox = z2.bounding_box()
        for _ in range(200):
            x = z.sample(1)[0]
            y = np.maximum(x, 0)
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= y[d] <= bbox[d, 1] + 0.01

    def test_relu_preserves_dimension(self):
        """Test ReLU preserves dimension."""
        z = Zonotope(
            center=np.array([1.0, -1.0, 0.5]),
            generators=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        relu = ReLUTransfer()
        z2 = relu.apply(z)
        assert z2.dimension == 3


class TestTanhTransfer:
    """Test Tanh abstract transfer function."""

    def test_small_input(self):
        """Test Tanh with small input (near-linear region)."""
        z = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[0.1, 0.0], [0.0, 0.1]])
        )
        tanh = TanhTransfer()
        z2 = tanh.apply(z)
        # For small inputs, tanh(x) ≈ x
        np.testing.assert_allclose(z2.center, [0.0, 0.0], atol=0.05)

    def test_large_input(self):
        """Test Tanh with large input (saturating region)."""
        z = Zonotope(
            center=np.array([10.0, -10.0]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5]])
        )
        tanh = TanhTransfer()
        z2 = tanh.apply(z)
        bbox = z2.bounding_box()
        assert bbox[0, 0] >= 0.5
        assert bbox[0, 1] <= 1.01
        assert bbox[1, 0] >= -1.01
        assert bbox[1, 1] <= -0.5

    def test_soundness_tanh(self):
        """Test soundness of Tanh abstract transformer."""
        np.random.seed(42)
        z = Zonotope(
            center=np.array([0.5, -0.3]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5], [0.2, 0.2]])
        )
        tanh = TanhTransfer()
        z2 = tanh.apply(z)
        bbox = z2.bounding_box()
        for _ in range(200):
            x = z.sample(1)[0]
            y = np.tanh(x)
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= y[d] <= bbox[d, 1] + 0.01

    def test_tanh_preserves_dimension(self):
        """Test Tanh preserves dimension."""
        z = Zonotope(
            center=np.array([1.0, -1.0, 0.5]),
            generators=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        tanh = TanhTransfer()
        z2 = tanh.apply(z)
        assert z2.dimension == 3


class TestTransferComposition:
    """Test composing transfer functions."""

    def test_linear_relu_chain(self):
        """Test linear -> ReLU -> linear chain."""
        z = Zonotope(
            center=np.array([1.0, -0.5]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5]])
        )
        W1 = np.array([[1.0, -0.5], [0.3, 1.0], [-0.2, 0.8]])
        b1 = np.array([0.1, -0.1, 0.0])
        W2 = np.array([[0.5, -0.3, 0.7], [0.2, 0.6, -0.4]])
        b2 = np.array([0.0, 0.1])

        lt1 = LinearTransfer(weight=W1, bias=b1)
        relu = ReLUTransfer()
        lt2 = LinearTransfer(weight=W2, bias=b2)

        z1 = lt1.apply(z)
        z2 = relu.apply(z1)
        z3 = lt2.apply(z2)

        assert z3.dimension == 2
        bbox = z3.bounding_box()
        assert np.all(np.isfinite(bbox))

    def test_chain_soundness(self):
        """Test soundness of transfer chain."""
        np.random.seed(42)
        z = Zonotope(
            center=np.array([0.5, 0.5]),
            generators=np.array([[0.3, 0.0], [0.0, 0.3]])
        )
        W1 = np.array([[1.0, -0.5], [0.3, 1.0]])
        b1 = np.array([0.1, -0.1])
        lt1 = LinearTransfer(weight=W1, bias=b1)
        relu = ReLUTransfer()

        z1 = lt1.apply(z)
        z2 = relu.apply(z1)
        bbox = z2.bounding_box()

        for _ in range(200):
            x = z.sample(1)[0]
            h = W1 @ x + b1
            y = np.maximum(h, 0)
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= y[d] <= bbox[d, 1] + 0.01
