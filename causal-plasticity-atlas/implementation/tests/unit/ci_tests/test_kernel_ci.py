"""Unit tests for cpa.ci_tests.kernel_ci."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from cpa.ci_tests.kernel_ci import (
    HSICTest,
    KCIResult,
    KernelCITest,
    _center_kernel_matrix,
    _compute_kernel_matrix,
    _median_bandwidth,
    _standardize,
    _unbiased_hsic,
)


# ── helpers ─────────────────────────────────────────────────────────

def _independent_data(n: int = 200, seed: int = 42):
    """Two independent Gaussian vectors."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 1)), rng.standard_normal((n, 1))


def _linear_dependent_data(n: int = 200, seed: int = 42):
    """Y = 2*X + noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1))
    y = 2.0 * x + 0.3 * rng.standard_normal((n, 1))
    return x, y


def _nonlinear_dependent_data(n: int = 300, seed: int = 42):
    """Y = X^2 + noise (nonlinear dependence)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1))
    y = x ** 2 + 0.2 * rng.standard_normal((n, 1))
    return x, y


def _confounded_triple(n: int = 300, seed: int = 42):
    """Z -> X, Z -> Y; X ⊥ Y | Z."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x = 0.8 * z + 0.3 * rng.standard_normal(n)
    y = 0.8 * z + 0.3 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


def _chain_triple(n: int = 300, seed: int = 42):
    """X -> Z -> Y; X ⊥ Y | Z."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = 0.9 * x + 0.2 * rng.standard_normal(n)
    y = 0.9 * z + 0.2 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


# ── _median_bandwidth tests ────────────────────────────────────────

class TestMedianBandwidth:
    def test_positive_bandwidth(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        bw = _median_bandwidth(X)
        assert bw > 0

    def test_single_point_returns_one(self):
        X = np.array([[1.0, 2.0]])
        assert _median_bandwidth(X) == 1.0

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bw = _median_bandwidth(X)
        assert bw > 0

    def test_constant_data_returns_one(self):
        X = np.ones((10, 2))
        bw = _median_bandwidth(X)
        assert bw == 1.0

    def test_spread_data_larger_bandwidth(self):
        rng = np.random.default_rng(42)
        X_narrow = rng.standard_normal((100, 1)) * 0.1
        X_wide = rng.standard_normal((100, 1)) * 10.0
        assert _median_bandwidth(X_wide) > _median_bandwidth(X_narrow)


# ── _compute_kernel_matrix tests ───────────────────────────────────

class TestComputeKernelMatrix:
    def test_rbf_shape_and_symmetry(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        K = _compute_kernel_matrix(X, kernel="rbf")
        assert K.shape == (20, 20)
        assert_array_almost_equal(K, K.T)

    def test_rbf_diagonal_ones(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        K = _compute_kernel_matrix(X, kernel="rbf")
        assert_array_almost_equal(np.diag(K), np.ones(20))

    def test_rbf_values_in_01(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        K = _compute_kernel_matrix(X, kernel="rbf")
        assert np.all(K >= 0.0)
        assert np.all(K <= 1.0 + 1e-10)

    def test_polynomial_kernel(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 2))
        K = _compute_kernel_matrix(X, kernel="polynomial", degree=3)
        assert K.shape == (15, 15)
        assert_array_almost_equal(K, K.T)

    def test_linear_kernel(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        K = _compute_kernel_matrix(X, kernel="linear")
        expected = X @ X.T
        assert_array_almost_equal(K, expected)

    def test_unknown_kernel_raises(self):
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="Unknown kernel"):
            _compute_kernel_matrix(X, kernel="invalid")

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0])
        K = _compute_kernel_matrix(X, kernel="rbf")
        assert K.shape == (3, 3)

    def test_custom_bandwidth(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 1))
        K1 = _compute_kernel_matrix(X, kernel="rbf", bandwidth=0.5)
        K2 = _compute_kernel_matrix(X, kernel="rbf", bandwidth=5.0)
        # Smaller bandwidth => sharper kernel => more variation
        assert np.std(K1) > np.std(K2)


# ── _center_kernel_matrix tests ────────────────────────────────────

class TestCenterKernelMatrix:
    def test_centered_row_col_means_zero(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        K = _compute_kernel_matrix(X, kernel="rbf")
        Kc = _center_kernel_matrix(K)
        assert_array_almost_equal(Kc.mean(axis=0), np.zeros(20), decimal=10)
        assert_array_almost_equal(Kc.mean(axis=1), np.zeros(20), decimal=10)

    def test_centered_symmetry(self):
        rng = np.random.default_rng(42)
        K = rng.standard_normal((10, 10))
        K = K @ K.T
        Kc = _center_kernel_matrix(K)
        assert_array_almost_equal(Kc, Kc.T)

    def test_double_centering_idempotent(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 2))
        K = _compute_kernel_matrix(X, kernel="rbf")
        Kc1 = _center_kernel_matrix(K)
        Kc2 = _center_kernel_matrix(Kc1)
        assert_array_almost_equal(Kc1, Kc2, decimal=10)


# ── _standardize tests ─────────────────────────────────────────────

class TestStandardize:
    def test_zero_mean_unit_var(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3)) * 5.0 + 10.0
        Xs = _standardize(X)
        assert_array_almost_equal(Xs.mean(axis=0), np.zeros(3), decimal=10)
        assert_array_almost_equal(Xs.std(axis=0), np.ones(3), decimal=10)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Xs = _standardize(X)
        assert Xs.shape == (5, 1)


# ── HSICTest tests ─────────────────────────────────────────────────

class TestHSICTest:
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HSICTest(alpha=0.0)

    def test_independent_data_high_pvalue(self):
        x, y = _independent_data(n=100, seed=42)
        hsic = HSICTest(alpha=0.05, n_permutations=200)
        stat, pval = hsic.test(x, y)
        assert pval > 0.05

    def test_linear_dependent_low_pvalue(self):
        x, y = _linear_dependent_data(n=100, seed=42)
        hsic = HSICTest(alpha=0.05, n_permutations=200)
        stat, pval = hsic.test(x, y)
        assert pval < 0.05

    def test_nonlinear_dependent_detected(self):
        x, y = _nonlinear_dependent_data(n=200, seed=42)
        hsic = HSICTest(alpha=0.05, n_permutations=500)
        stat, pval = hsic.test(x, y)
        assert pval < 0.1  # nonlinear dependence should be detectable

    def test_hsic_statistic_nonnegative(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 1))
        Y = rng.standard_normal((30, 1))
        Kx = _compute_kernel_matrix(X, kernel="rbf")
        Ky = _compute_kernel_matrix(Y, kernel="rbf")
        hsic = HSICTest()
        stat = hsic.hsic_statistic(Kx, Ky)
        assert stat >= -1e-10  # biased HSIC should be non-negative

    def test_unbiased_hsic_small_sample(self):
        hsic = HSICTest()
        Kx = np.eye(3)
        Ky = np.eye(3)
        assert hsic.unbiased_hsic(Kx, Ky) == 0.0

    def test_too_few_samples(self):
        x = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        y = np.array([4.0, 5.0, 6.0]).reshape(-1, 1)
        hsic = HSICTest()
        stat, pval = hsic.test(x, y)
        assert pval == 1.0

    def test_mismatched_lengths_raises(self):
        hsic = HSICTest()
        with pytest.raises(ValueError, match="same number"):
            hsic.test(np.ones((10, 1)), np.ones((8, 1)))

    def test_gamma_approximation_runs(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 1))
        Y = rng.standard_normal((50, 1))
        Kx = _compute_kernel_matrix(X, kernel="rbf")
        Ky = _compute_kernel_matrix(Y, kernel="rbf")
        hsic = HSICTest()
        stat = hsic.hsic_statistic(Kx, Ky)
        pval = hsic.gamma_approximation(stat, Kx, Ky)
        assert 0.0 <= pval <= 1.0


# ── KernelCITest tests ─────────────────────────────────────────────

class TestKernelCITest:
    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            KernelCITest(alpha=1.5)

    def test_not_2d_raises(self):
        kci = KernelCITest()
        with pytest.raises(ValueError, match="2-D"):
            kci.test(np.ones(10), 0, 1)

    def test_index_out_of_range_raises(self):
        kci = KernelCITest()
        data = np.ones((10, 3))
        with pytest.raises(IndexError):
            kci.test(data, 0, 5)

    def test_few_samples_returns_trivial(self):
        kci = KernelCITest()
        data = np.ones((4, 3))
        stat, pval = kci.test(data, 0, 1)
        assert pval == 1.0

    def test_no_conditioning_reduces_to_hsic(self):
        """With no conditioning set, KCI should behave like HSIC."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        kci = KernelCITest(alpha=0.05, n_bootstrap=100)
        stat, pval = kci.test(data, 0, 1)
        assert 0 <= pval <= 1

    @pytest.mark.slow
    def test_conditional_independence_confounded(self):
        """X ⊥ Y | Z for confounded triple."""
        data = _confounded_triple(n=200, seed=42)
        kci = KernelCITest(alpha=0.05, n_bootstrap=200)
        stat, pval = kci.test(data, 0, 1, conditioning_set={2})
        assert pval > 0.01

    @pytest.mark.slow
    def test_marginal_dependence_confounded(self):
        """X and Y are marginally dependent (confounded)."""
        data = _confounded_triple(n=200, seed=42)
        kci = KernelCITest(alpha=0.05, n_bootstrap=200)
        stat, pval = kci.test(data, 0, 1)
        assert pval < 0.1

    def test_test_full_returns_kci_result(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 3))
        kci = KernelCITest(alpha=0.05, n_bootstrap=50)
        result = kci.test_full(data, 0, 1)
        assert isinstance(result, KCIResult)
        assert result.method == "kci"

    def test_compute_test_statistic_finite(self):
        rng = np.random.default_rng(42)
        n = 30
        Kx = _compute_kernel_matrix(rng.standard_normal((n, 1)), "rbf")
        Ky = _compute_kernel_matrix(rng.standard_normal((n, 1)), "rbf")
        Kz = _compute_kernel_matrix(rng.standard_normal((n, 1)), "rbf")
        kci = KernelCITest()
        stat = kci.compute_test_statistic(Kx, Ky, Kz)
        assert np.isfinite(stat)

    @pytest.mark.slow
    def test_linear_data_with_conditioning(self):
        """X -> Z -> Y chain with linear relationships."""
        data = _chain_triple(n=200, seed=42)
        kci = KernelCITest(alpha=0.05, n_bootstrap=200)
        stat, pval = kci.test(data, 0, 1, conditioning_set={2})
        assert pval > 0.01
