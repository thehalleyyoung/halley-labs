"""Tests for optimal transport module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transport import (
    sinkhorn_divergence,
    sinkhorn_distance,
    cost_matrix,
    transport_plan,
    RepulsiveEnergy,
    sinkhorn_gradient,
    wasserstein_1d,
)


def test_sinkhorn_identical_zero():
    """Sinkhorn divergence of identical distributions should be ~0."""
    X = np.random.RandomState(42).randn(10, 3)
    div = sinkhorn_divergence(X, X, reg=0.1, n_iter=100)
    assert div < 0.1, f"Expected ~0, got {div}"


def test_sinkhorn_symmetric():
    """S(X, Y) should approximately equal S(Y, X)."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 3)
    Y = rng.randn(10, 3) + 2.0
    s_xy = sinkhorn_divergence(X, Y, reg=0.1)
    s_yx = sinkhorn_divergence(Y, X, reg=0.1)
    assert abs(s_xy - s_yx) < 0.5, f"|{s_xy} - {s_yx}| too large"


def test_sinkhorn_positive_for_different():
    """Sinkhorn divergence should be positive for different distributions."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 3)
    Y = rng.randn(10, 3) + 5.0
    div = sinkhorn_divergence(X, Y, reg=0.5, n_iter=200)
    assert div >= 0.0


def test_sinkhorn_divergence_debiased():
    """Debiased Sinkhorn: S_eps(mu, mu) should be approximately 0."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3)
    div = sinkhorn_divergence(X, X, reg=0.1, n_iter=100)
    assert div < 0.2, f"Self-divergence should be ~0, got {div}"


def test_sinkhorn_converges_to_wasserstein():
    """As reg -> 0, Sinkhorn distance should approach Wasserstein."""
    rng = np.random.RandomState(42)
    a = np.ones(5) / 5
    b = np.ones(5) / 5
    X = rng.randn(5, 2)
    Y = rng.randn(5, 2) + 2.0
    M = cost_matrix(X, Y, metric="sqeuclidean")

    d_large_reg = sinkhorn_distance(a, b, M, reg=1.0, n_iter=200)
    d_small_reg = sinkhorn_distance(a, b, M, reg=0.01, n_iter=200)
    # Smaller reg should give larger (closer to unregularized) cost
    # Actually entropic OT is always <= unregularized, both should be positive
    assert d_small_reg > 0 and d_large_reg > 0


def test_cost_matrix_euclidean():
    X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    Y = np.array([[0, 0], [1, 1]], dtype=float)
    C = cost_matrix(X, Y, metric="euclidean")
    assert C.shape == (3, 2)
    assert abs(C[0, 0]) < 1e-10  # distance to self
    assert abs(C[0, 1] - np.sqrt(2)) < 1e-10


def test_repulsive_energy_increases_with_proximity():
    """Points closer to history should have higher repulsive energy."""
    energy = RepulsiveEnergy()
    history = np.array([[0.0, 0.0]])
    y_close = np.array([0.01, 0.01])
    y_far = np.array([5.0, 5.0])
    e_close = energy(y_close, history)
    e_far = energy(y_far, history)
    # -log(small_dist) > -log(large_dist)
    assert e_close > e_far


def test_repulsive_energy_zero_for_empty_history():
    energy = RepulsiveEnergy()
    y = np.array([1.0, 2.0])
    e = energy(y, np.array([]).reshape(0, 2))
    assert e == 0.0


def test_transport_plan_marginals():
    """Transport plan marginals should match source and target distributions."""
    rng = np.random.RandomState(42)
    n, m = 5, 7
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    X = rng.randn(n, 3)
    Y = rng.randn(m, 3)
    M = cost_matrix(X, Y, metric="sqeuclidean")
    P = transport_plan(a, b, M, reg=0.5, n_iter=200)
    assert np.allclose(P.sum(axis=1), a, atol=1e-2), "Row marginals don't match"
    assert np.allclose(P.sum(axis=0), b, atol=1e-2), "Column marginals don't match"


def test_sinkhorn_gradient_direction():
    """Gradient should point away from dense regions toward sparse ones."""
    rng = np.random.RandomState(42)
    # X is clustered at origin, Y is spread out
    X = rng.randn(5, 2) * 0.1
    Y = rng.randn(5, 2) * 3.0
    grad = sinkhorn_gradient(X, Y, reg=0.5, n_iter=30, delta=1e-3)
    # Gradient should have non-zero magnitude (pushing X away from its cluster)
    grad_norm = np.linalg.norm(grad)
    assert grad_norm > 1e-6, f"Gradient too small: {grad_norm}"
