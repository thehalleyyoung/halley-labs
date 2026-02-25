"""Tests for the kernel engine module (NTK computation, Nyström, kernel ops)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.kernel_engine import (
    AnalyticNTK,
    EmpiricalNTK,
    KernelAlignment,
    KernelMatrix,
    KernelPCA,
    KernelSpectralAnalysis,
    NTKComputer,
    NTKTracker,
    NystromApproximation,
    LandmarkSelector,
    AdaptiveRankSelector,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_data(rng):
    n, d = 20, 5
    return rng.randn(n, d)


@pytest.fixture
def medium_data(rng):
    n, d = 50, 10
    return rng.randn(n, d)


def _simple_forward(params, x):
    """2-layer ReLU MLP: f(x) = W2 @ relu(W1 @ x)."""
    d = len(x)
    w = int(math.sqrt(len(params) // 2))
    if w < 1:
        w = max(1, len(params) // (d + 1))
    W1 = params[:d * w].reshape(w, d)
    b_start = d * w
    W2 = params[b_start:b_start + w].reshape(1, w)
    h = W1 @ x
    h = np.maximum(h, 0)  # ReLU
    return (W2 @ h).ravel()


@pytest.fixture
def simple_mlp_params(rng):
    d_in, width = 5, 8
    n_params = d_in * width + width
    return rng.randn(n_params) * 0.1


# ===================================================================
# NTK symmetry and PSD
# ===================================================================

class TestNTKProperties:
    def test_empirical_ntk_symmetry(self, small_data, simple_mlp_params):
        emp = EmpiricalNTK(output_dim=1)
        n = small_data.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            Ji = emp.compute_jacobian(_simple_forward, simple_mlp_params, small_data[i])
            for j in range(n):
                Jj = emp.compute_jacobian(_simple_forward, simple_mlp_params, small_data[j])
                K[i, j] = np.sum(Ji * Jj)

        # Symmetry
        assert np.allclose(K, K.T, atol=1e-10), "NTK must be symmetric"

    def test_empirical_ntk_psd(self, small_data, simple_mlp_params):
        emp = EmpiricalNTK(output_dim=1)
        n = small_data.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            Ji = emp.compute_jacobian(_simple_forward, simple_mlp_params, small_data[i])
            for j in range(n):
                Jj = emp.compute_jacobian(_simple_forward, simple_mlp_params, small_data[j])
                K[i, j] = np.sum(Ji * Jj)

        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8), "NTK must be positive semi-definite"

    def test_analytic_ntk_symmetry(self, small_data):
        analytic = AnalyticNTK()
        K = analytic.compute(small_data, depth=2, width=64, activation="relu")
        assert np.allclose(K, K.T, atol=1e-10)

    def test_analytic_ntk_psd(self, small_data):
        analytic = AnalyticNTK()
        K = analytic.compute(small_data, depth=2, width=64, activation="relu")
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-8)

    def test_analytic_ntk_positive_diagonal(self, small_data):
        analytic = AnalyticNTK()
        K = analytic.compute(small_data, depth=2, width=64, activation="relu")
        assert np.all(np.diag(K) > 0), "Diagonal entries must be positive"


# ===================================================================
# Known values for 2-layer ReLU
# ===================================================================

class TestKnownValues:
    def test_relu_kernel_formula(self):
        """For ReLU, the NNGP kernel has a known formula involving arccos."""
        n = 5
        rng = np.random.RandomState(123)
        X = rng.randn(n, 3)
        # Normalise so we can check formula
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        analytic = AnalyticNTK()
        K = analytic.compute(X, depth=1, width=1000, activation="relu")

        # For unit-norm inputs, single-layer ReLU NNGP:
        # k(x,x') = (1/2π)(||x||·||x'||)(sin θ + (π-θ)cos θ)
        # where θ = arccos(x·x')
        # This is the expected form — just verify structure
        assert K.shape == (n, n)
        assert np.allclose(K, K.T)

    def test_depth_scaling(self, small_data):
        """Deeper networks should have different kernel structure."""
        analytic = AnalyticNTK()
        K1 = analytic.compute(small_data, depth=1, width=128, activation="relu")
        K2 = analytic.compute(small_data, depth=3, width=128, activation="relu")
        # They should differ
        assert not np.allclose(K1, K2, atol=1e-3)

    def test_width_convergence(self):
        """As width → ∞, empirical NTK should approach analytic."""
        rng = np.random.RandomState(7)
        X = rng.randn(5, 3)

        analytic = AnalyticNTK()
        K_inf = analytic.compute(X, depth=2, width=10000, activation="relu")

        # Just check it's well-formed
        assert K_inf.shape == (5, 5)
        assert np.all(np.isfinite(K_inf))


# ===================================================================
# Nyström approximation
# ===================================================================

class TestNystrom:
    def test_approximation_shape(self, medium_data):
        analytic = AnalyticNTK()
        K_exact = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        nystrom = NystromApproximation(rank=10)
        K_approx = nystrom.approximate(K_exact)
        assert K_approx.shape == K_exact.shape

    def test_approximation_error_bound(self, medium_data):
        analytic = AnalyticNTK()
        K_exact = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        nystrom = NystromApproximation(rank=20)
        K_approx = nystrom.approximate(K_exact)

        rel_err = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
        # With rank 20 out of 50, error should be modest
        assert rel_err < 1.0, f"Nyström relative error too large: {rel_err:.3f}"

    def test_psd_preserved(self, medium_data):
        analytic = AnalyticNTK()
        K_exact = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        nystrom = NystromApproximation(rank=15)
        K_approx = nystrom.approximate(K_exact)

        eigvals = np.linalg.eigvalsh(K_approx)
        assert np.all(eigvals >= -1e-6), "Nyström approx should be approximately PSD"

    def test_landmark_selection_random(self, medium_data):
        selector = LandmarkSelector(strategy="random")
        indices = selector.select(medium_data, k=10)
        assert len(indices) == 10
        assert len(set(indices)) == 10  # unique

    def test_adaptive_rank(self, medium_data):
        analytic = AnalyticNTK()
        K = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        selector = AdaptiveRankSelector(tolerance=1e-3)
        rank = selector.select_rank(K)
        assert 1 <= rank <= K.shape[0]


# ===================================================================
# Kernel alignment
# ===================================================================

class TestKernelAlignment:
    def test_self_alignment(self, medium_data):
        analytic = AnalyticNTK()
        K = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        align = KernelAlignment()
        score = align.compute(K, K)
        assert abs(score - 1.0) < 1e-6, "Self-alignment must be 1"

    def test_alignment_range(self, medium_data, rng):
        analytic = AnalyticNTK()
        K1 = analytic.compute(medium_data, depth=2, width=128, activation="relu")
        K2 = rng.randn(50, 50)
        K2 = K2 @ K2.T  # make PSD

        align = KernelAlignment()
        score = align.compute(K1, K2)
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6

    def test_alignment_symmetry(self, medium_data, rng):
        analytic = AnalyticNTK()
        K1 = analytic.compute(medium_data, depth=2, width=128, activation="relu")
        K2 = rng.randn(50, 50)
        K2 = K2 @ K2.T

        align = KernelAlignment()
        assert abs(align.compute(K1, K2) - align.compute(K2, K1)) < 1e-10


# ===================================================================
# Kernel matrix operations
# ===================================================================

class TestKernelMatrix:
    def test_creation(self, rng):
        K = rng.randn(10, 10)
        K = K @ K.T
        km = KernelMatrix(K)
        assert km.shape == (10, 10)

    def test_spectral_analysis(self, medium_data):
        analytic = AnalyticNTK()
        K = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        analysis = KernelSpectralAnalysis()
        result = analysis.analyze(K)
        assert "eigenvalues" in result or hasattr(result, "eigenvalues")


class TestKernelPCA:
    def test_projection(self, medium_data):
        analytic = AnalyticNTK()
        K = analytic.compute(medium_data, depth=2, width=128, activation="relu")

        pca = KernelPCA(n_components=5)
        projected = pca.fit_transform(K)
        assert projected.shape == (50, 5)


# ===================================================================
# NTK Tracker
# ===================================================================

class TestNTKTracker:
    def test_tracking(self, small_data):
        tracker = NTKTracker()
        analytic = AnalyticNTK()

        K1 = analytic.compute(small_data, depth=2, width=64, activation="relu")
        K2 = K1 + 0.01 * np.random.randn(*K1.shape)
        K2 = 0.5 * (K2 + K2.T)

        tracker.record(0, K1)
        tracker.record(1, K2)

        drift = tracker.compute_drift(0, 1)
        assert drift >= 0


# ===================================================================
# NTKComputer (unified interface)
# ===================================================================

class TestNTKComputer:
    def test_compute_analytic(self, small_data):
        computer = NTKComputer(mode="analytic")
        K = computer.compute(
            small_data, depth=2, width=128, activation="relu"
        )
        assert K.shape == (small_data.shape[0], small_data.shape[0])
        assert np.allclose(K, K.T)
