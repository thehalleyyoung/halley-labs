"""Tests for kernel functions."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import (
    RBFKernel,
    MaternKernel,
    CosineKernel,
    PolynomialKernel,
    SpectralKernel,
    AdaptiveRBFKernel,
    ManifoldAdaptiveKernel,
    QualityDiversityKernel,
)
from src.utils import psd_check


def _make_data(n=10, d=4, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d)


def test_rbf_psd():
    X = _make_data()
    kernel = RBFKernel(bandwidth=1.0)
    G = kernel.gram_matrix(X)
    assert psd_check(G), "RBF Gram matrix not PSD"


def test_rbf_identical_is_one():
    kernel = RBFKernel(bandwidth=1.0)
    x = np.array([1.0, 2.0, 3.0])
    assert abs(kernel.evaluate(x, x) - 1.0) < 1e-10


def test_matern_psd():
    X = _make_data()
    kernel = MaternKernel(nu=1.5, length_scale=1.0)
    G = kernel.gram_matrix(X)
    assert psd_check(G), "Matern Gram matrix not PSD"


def test_cosine_psd():
    X = _make_data()
    kernel = CosineKernel()
    G = kernel.gram_matrix(X)
    assert psd_check(G), "Cosine Gram matrix not PSD"


def test_polynomial_psd():
    X = _make_data()
    kernel = PolynomialKernel(degree=2, c=1.0)
    G = kernel.gram_matrix(X)
    assert psd_check(G), "Polynomial Gram matrix not PSD"


def test_adaptive_rbf_updates_bandwidth():
    kernel = AdaptiveRBFKernel(initial_bandwidth=1.0)
    X = _make_data(n=20, d=4, seed=10)
    old_bw = kernel.bandwidth
    kernel.update(X)
    # Bandwidth should have changed
    assert kernel.bandwidth != old_bw, "Bandwidth did not update"
    assert kernel.bandwidth > 0


def test_manifold_adaptive_local_pca():
    X = _make_data(n=20, d=4, seed=11)
    kernel = ManifoldAdaptiveKernel(bandwidth=1.0, n_neighbors=5)
    G = kernel.gram_matrix(X)
    # Should be symmetric
    assert np.allclose(G, G.T, atol=1e-10)
    # Diagonal should be 1
    assert np.allclose(np.diag(G), 1.0, atol=1e-10)


def test_quality_diversity_kernel():
    X = _make_data(n=5, d=3, seed=12)
    base = RBFKernel(bandwidth=1.0)
    qd = QualityDiversityKernel(base)
    qualities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    qd.set_qualities(qualities)
    G = qd.gram_matrix(X)
    G_base = base.gram_matrix(X)
    # L_ij = q_i * q_j * K_ij
    expected = G_base * np.outer(qualities, qualities)
    assert np.allclose(G, expected, atol=1e-10)


def test_gram_matrix_symmetric():
    X = _make_data()
    for KernelClass in [RBFKernel, MaternKernel, CosineKernel]:
        kernel = KernelClass() if KernelClass != MaternKernel else MaternKernel(nu=2.5)
        G = kernel.gram_matrix(X)
        assert np.allclose(G, G.T, atol=1e-10), f"{KernelClass.__name__} not symmetric"


def test_spectral_approximation_close():
    """Spectral kernel should approximate RBF."""
    X = _make_data(n=5, d=3, seed=13)
    rbf = RBFKernel(bandwidth=1.0)
    spec = SpectralKernel(n_components=500, bandwidth=1.0, seed=42)
    G_rbf = rbf.gram_matrix(X)
    G_spec = spec.gram_matrix(X)
    # Should be reasonably close (RFF approximation)
    assert np.allclose(G_rbf, G_spec, atol=0.3), \
        f"Max diff: {np.max(np.abs(G_rbf - G_spec))}"
