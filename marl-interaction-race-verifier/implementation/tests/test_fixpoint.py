"""Tests for fixpoint computation engine."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import (
    FixpointEngine,
    ConvergenceChecker,
    BoundedAscendingChain,
)
from marace.abstract.transfer import LinearTransfer, ReLUTransfer


class TestConvergenceChecker:
    """Test convergence checking."""

    def test_converged_identical(self):
        """Test convergence for identical zonotopes."""
        z1 = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        z2 = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        checker = ConvergenceChecker(threshold=1e-6, patience=1)
        converged, dist = checker.check(z1, z2)
        assert converged

    def test_not_converged(self):
        """Test non-convergence for different zonotopes."""
        z1 = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        z2 = Zonotope(
            center=np.array([5.0, 5.0]),
            generators=np.array([[2.0, 0.0], [0.0, 2.0]])
        )
        checker = ConvergenceChecker(threshold=1e-6, patience=1)
        converged, dist = checker.check(z1, z2)
        assert not converged

    def test_near_convergence(self):
        """Test near-convergence with small threshold."""
        z1 = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0]])
        )
        z2 = Zonotope(
            center=np.array([1.001, 2.001]),
            generators=np.array([[1.001, 0.001]])
        )
        checker = ConvergenceChecker(threshold=0.01, patience=1)
        converged, dist = checker.check(z1, z2)
        assert converged


class TestBoundedAscendingChain:
    """Test bounded ascending chain tracking."""

    def test_chain_creation(self):
        """Test creating a chain."""
        chain = BoundedAscendingChain(max_length=100)
        z = Zonotope(
            center=np.array([0.0]),
            generators=np.array([[1.0]])
        )
        chain.add(z)
        assert chain.length == 1

    def test_chain_growth(self):
        """Test chain grows monotonically."""
        chain = BoundedAscendingChain(max_length=50)
        for i in range(10):
            z = Zonotope(
                center=np.array([0.0]),
                generators=np.array([[float(i + 1)]])
            )
            chain.add(z)
        assert chain.length == 10

    def test_chain_max_length(self):
        """Test chain respects max length."""
        chain = BoundedAscendingChain(max_length=5)
        for i in range(10):
            z = Zonotope(
                center=np.array([0.0]),
                generators=np.array([[float(i + 1)]])
            )
            chain.add(z)
            if chain.terminated:
                break
        assert chain.length <= 5


class TestFixpointEngine:
    """Test fixpoint computation."""

    def test_stable_fixpoint(self):
        """Test fixpoint for contractive transformation."""
        def contractive_transfer(z: Zonotope) -> Zonotope:
            """Apply a contractive affine transformation."""
            W = np.array([[0.5, 0.0], [0.0, 0.5]])
            b = np.array([0.5, 0.5])
            return z.affine_transform(W, b)

        engine = FixpointEngine(
            transfer_fn=contractive_transfer,
            max_iterations=100,
            convergence_threshold=1e-4,
        )

        z0 = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[2.0, 0.0], [0.0, 2.0]])
        )
        result = engine.compute(z0)
        assert result.converged
        assert result.iterations < 100
        # Fixed point of x -> 0.5x + 0.5 is x = 1.0
        np.testing.assert_allclose(result.element.center, [1.0, 1.0], atol=0.2)

    def test_linear_fixpoint(self):
        """Test fixpoint for linear system."""
        def stable_linear(z: Zonotope) -> Zonotope:
            W = np.array([[0.8, 0.1], [-0.1, 0.8]])
            b = np.array([0.1, 0.1])
            return z.affine_transform(W, b)

        engine = FixpointEngine(
            transfer_fn=stable_linear,
            max_iterations=200,
            convergence_threshold=1e-3,
        )

        z0 = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        result = engine.compute(z0)
        assert result.converged or result.iterations == 200

    def test_fixpoint_with_widening(self):
        """Test fixpoint with widening for expanding system."""
        step = [0]
        def expanding_then_stable(z: Zonotope) -> Zonotope:
            step[0] += 1
            if step[0] < 5:
                W = np.array([[1.1, 0.0], [0.0, 1.1]])
            else:
                W = np.array([[0.9, 0.0], [0.0, 0.9]])
            b = np.array([0.1, 0.1])
            return z.affine_transform(W, b)

        engine = FixpointEngine(
            transfer_fn=expanding_then_stable,
            max_iterations=50,
            convergence_threshold=1e-3,
            delay_widening=3,
        )

        z0 = Zonotope(
            center=np.array([0.0, 0.0]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5]])
        )
        result = engine.compute(z0)
        # Should either converge or reach max iterations
        assert result.iterations <= 50

    def test_fixpoint_result_properties(self):
        """Test fixpoint result has expected properties."""
        def identity_transfer(z: Zonotope) -> Zonotope:
            return z

        engine = FixpointEngine(
            transfer_fn=identity_transfer,
            max_iterations=10,
            convergence_threshold=1e-4,
        )

        z0 = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[0.5, 0.0]])
        )
        result = engine.compute(z0)
        assert result.converged
        assert result.iterations <= 2
        np.testing.assert_allclose(result.element.center, z0.center)
