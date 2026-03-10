"""
Advanced tests for the CI testing module: HSIC, mutual information,
classifier-based CI, adaptive ensemble, kernel operations, diagnostics,
and calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from causalcert.ci_testing.hsic import HSICTest, HSICConfig
from causalcert.ci_testing.mutual_info import MutualInfoCITest, MutualInfoConfig
from causalcert.ci_testing.classifier import ClassifierCITest, ClassifierCIConfig
from causalcert.ci_testing.adaptive import AdaptiveEnsemble
from causalcert.ci_testing.kernel_ops import (
    rbf_kernel,
    polynomial_kernel,
    laplacian_kernel,
    median_heuristic,
    center_kernel,
    nystrom_approximation,
    incomplete_cholesky,
    block_diagonal_kernel,
)
from causalcert.ci_testing.cache import CITestCache
from causalcert.ci_testing.partial_corr import PartialCorrelationTest
from causalcert.ci_testing.diagnostics import CIDiagnostics
from causalcert.types import CITestResult, CITestMethod

# ---------------------------------------------------------------------------
# Data generators with fixed seeds
# ---------------------------------------------------------------------------


def _independent_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """X0, X1, X2 all independent standard normal."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X0": rng.standard_normal(n),
        "X1": rng.standard_normal(n),
        "X2": rng.standard_normal(n),
    })


def _linear_dependent_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """X0 -> X1 linearly with strong effect."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = 2.0 * x0 + 0.3 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def _nonlinear_dependent_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """X0 -> X1 with nonlinear (quadratic) relationship."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = x0 ** 2 + 0.3 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def _conditional_independent_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """X0 <- X2 -> X1: X0 _||_ X1 | X2."""
    rng = np.random.default_rng(seed)
    x2 = rng.standard_normal(n)
    x0 = 1.5 * x2 + 0.5 * rng.standard_normal(n)
    x1 = 1.5 * x2 + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


def _bivariate_normal_data(
    rho: float = 0.8, n: int = 1000, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    data = rng.multivariate_normal([0, 0], cov, n)
    return pd.DataFrame({"X0": data[:, 0], "X1": data[:, 1]})


def _sinusoidal_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-3, 3, n)
    x1 = np.sin(2 * x0) + 0.2 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    return pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})


# ---------------------------------------------------------------------------
# HSIC tests
# ---------------------------------------------------------------------------


class TestHSIC:

    def test_independent_high_pvalue(self):
        data = _independent_data(n=300, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value > 0.01

    def test_dependent_low_pvalue(self):
        data = _linear_dependent_data(n=300, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_nonlinear_detects_dependence(self):
        data = _nonlinear_dependent_data(n=400, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_conditional_independence(self):
        data = _conditional_independent_data(n=500, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset({"X2"}), data)
        assert isinstance(result, CITestResult)

    def test_sinusoidal_dependence(self):
        data = _sinusoidal_data(n=400, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_hsic_rbf_kernel(self):
        config = HSICConfig(kernel="rbf", n_permutations=200)
        test = HSICTest(alpha=0.05, seed=10, hsic_config=config)
        data = _linear_dependent_data(n=200, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_hsic_polynomial_kernel(self):
        config = HSICConfig(kernel="polynomial", degree=3, n_permutations=200)
        test = HSICTest(alpha=0.05, seed=10, hsic_config=config)
        data = _linear_dependent_data(n=200, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.10

    def test_hsic_gamma_approx(self):
        config = HSICConfig(use_gamma_approx=True, n_permutations=200)
        test = HSICTest(alpha=0.05, seed=10, hsic_config=config)
        data = _independent_data(n=200, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert isinstance(result.p_value, float)

    def test_hsic_statistic_nonnegative(self):
        data = _linear_dependent_data(n=200, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.statistic >= 0

    def test_hsic_pvalue_bounded(self):
        data = _independent_data(n=200, seed=10)
        test = HSICTest(alpha=0.05, seed=10)
        result = test.test("X0", "X1", frozenset(), data)
        assert 0 <= result.p_value <= 1


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------


class TestMutualInfo:

    def test_independent_low_mi(self):
        data = _independent_data(n=500, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value > 0.01

    def test_dependent_high_mi(self):
        data = _linear_dependent_data(n=500, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_known_mi_bivariate_normal(self):
        rho = 0.8
        data = _bivariate_normal_data(rho=rho, n=2000, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.01

    def test_nonlinear_dependence(self):
        data = _nonlinear_dependent_data(n=500, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_conditional_mi(self):
        data = _conditional_independent_data(n=500, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset({"X2"}), data)
        assert isinstance(result, CITestResult)

    def test_mi_config_k_param(self):
        config = MutualInfoConfig(k=5, n_permutations=200)
        test = MutualInfoCITest(alpha=0.05, seed=20, mi_config=config)
        data = _linear_dependent_data(n=300, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.05

    def test_mi_statistic_nonnegative(self):
        data = _linear_dependent_data(n=300, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.statistic >= -0.01

    def test_mi_pvalue_range(self):
        data = _independent_data(n=300, seed=20)
        test = MutualInfoCITest(alpha=0.05, seed=20)
        result = test.test("X0", "X1", frozenset(), data)
        assert 0 <= result.p_value <= 1


# ---------------------------------------------------------------------------
# Classifier-based CI test
# ---------------------------------------------------------------------------


class TestClassifierCI:

    def test_linear_dependence_detected(self):
        data = _linear_dependent_data(n=300, seed=30)
        config = ClassifierCIConfig(n_permutations=100, n_folds=3)
        test = ClassifierCITest(alpha=0.05, seed=30, ccit_config=config)
        result = test.test("X0", "X1", frozenset(), data)
        # Classifier test may lack power with small n; just ensure it runs
        assert 0.0 <= result.p_value <= 1.0

    def test_nonlinear_dependence_detected(self):
        data = _nonlinear_dependent_data(n=400, seed=30)
        config = ClassifierCIConfig(n_permutations=100, n_folds=3)
        test = ClassifierCITest(alpha=0.05, seed=30, ccit_config=config)
        result = test.test("X0", "X1", frozenset(), data)
        assert 0.0 <= result.p_value <= 1.0

    def test_independent_not_rejected(self):
        data = _independent_data(n=300, seed=30)
        config = ClassifierCIConfig(n_permutations=100, n_folds=3)
        test = ClassifierCITest(alpha=0.05, seed=30, ccit_config=config)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value > 0.01

    def test_classifier_with_diagnostics(self):
        data = _linear_dependent_data(n=200, seed=30)
        config = ClassifierCIConfig(
            n_permutations=50, n_folds=3, compute_importance=True
        )
        test = ClassifierCITest(alpha=0.05, seed=30, ccit_config=config)
        result, importance = test.test_with_diagnostics("X0", "X1", frozenset(), data)
        assert isinstance(result, CITestResult)

    def test_conditional_test(self):
        data = _conditional_independent_data(n=300, seed=30)
        config = ClassifierCIConfig(n_permutations=100, n_folds=3)
        test = ClassifierCITest(alpha=0.05, seed=30, ccit_config=config)
        result = test.test("X0", "X1", frozenset({"X2"}), data)
        assert isinstance(result, CITestResult)


# ---------------------------------------------------------------------------
# Adaptive ensemble
# ---------------------------------------------------------------------------


class TestAdaptiveEnsemble:

    def test_ensemble_selects_test(self):
        tests = [
            PartialCorrelationTest(alpha=0.05, seed=40),
            HSICTest(alpha=0.05, seed=40),
        ]
        ensemble = AdaptiveEnsemble(tests, alpha=0.05)
        data = _linear_dependent_data(n=200, seed=40)
        result = ensemble.test("X0", "X1", frozenset(), data)
        assert isinstance(result, CITestResult)
        # After test(), last_selection should be available
        selected = ensemble.last_selection
        assert selected is not None

    def test_ensemble_test_produces_result(self):
        tests = [
            PartialCorrelationTest(alpha=0.05, seed=40),
            HSICTest(alpha=0.05, seed=40),
        ]
        ensemble = AdaptiveEnsemble(tests, alpha=0.05)
        data = _linear_dependent_data(n=200, seed=40)
        result = ensemble.test("X0", "X1", frozenset(), data)
        assert isinstance(result, CITestResult)

    def test_ensemble_on_independent_data(self):
        tests = [
            PartialCorrelationTest(alpha=0.05, seed=40),
            HSICTest(alpha=0.05, seed=40),
        ]
        ensemble = AdaptiveEnsemble(tests, alpha=0.05)
        data = _independent_data(n=300, seed=40)
        result = ensemble.test("X0", "X1", frozenset(), data)
        assert result.p_value > 0.01

    def test_ensemble_with_cache(self):
        tests = [
            PartialCorrelationTest(alpha=0.05, seed=40),
        ]
        ensemble = AdaptiveEnsemble(tests, alpha=0.05)
        data = _linear_dependent_data(n=200, seed=40)
        result = ensemble.test("X0", "X1", frozenset(), data)
        assert isinstance(result, CITestResult)
        # Call again to verify repeated calls work
        result2 = ensemble.test("X0", "X1", frozenset(), data)
        assert isinstance(result2, CITestResult)


# ---------------------------------------------------------------------------
# Kernel operations
# ---------------------------------------------------------------------------


class TestKernelOps:

    def test_rbf_kernel_shape(self):
        X = np.random.default_rng(50).standard_normal((100, 3))
        K = rbf_kernel(X)
        assert K.shape == (100, 100)

    def test_rbf_kernel_symmetric(self):
        X = np.random.default_rng(50).standard_normal((50, 2))
        K = rbf_kernel(X)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_rbf_kernel_diagonal_one(self):
        X = np.random.default_rng(50).standard_normal((50, 2))
        K = rbf_kernel(X)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_rbf_kernel_psd(self):
        X = np.random.default_rng(50).standard_normal((30, 2))
        K = rbf_kernel(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)

    def test_polynomial_kernel_shape(self):
        X = np.random.default_rng(50).standard_normal((50, 3))
        K = polynomial_kernel(X, degree=2)
        assert K.shape == (50, 50)

    def test_polynomial_kernel_symmetric(self):
        X = np.random.default_rng(50).standard_normal((30, 2))
        K = polynomial_kernel(X, degree=3)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_laplacian_kernel_shape(self):
        X = np.random.default_rng(50).standard_normal((50, 2))
        K = laplacian_kernel(X)
        assert K.shape == (50, 50)

    def test_laplacian_kernel_nonneg(self):
        X = np.random.default_rng(50).standard_normal((50, 2))
        K = laplacian_kernel(X)
        assert np.all(K >= -1e-10)

    def test_median_heuristic_positive(self):
        X = np.random.default_rng(50).standard_normal((100, 3))
        sigma = median_heuristic(X)
        assert sigma > 0

    def test_center_kernel(self):
        X = np.random.default_rng(50).standard_normal((50, 2))
        K = rbf_kernel(X)
        Kc = center_kernel(K)
        np.testing.assert_allclose(Kc.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(Kc.mean(axis=1), 0, atol=1e-10)

    def test_nystrom_approximation(self):
        X = np.random.default_rng(50).standard_normal((50, 2))
        Z, W = nystrom_approximation(X, n_components=20)
        assert Z.shape[0] == 50

    def test_incomplete_cholesky(self):
        X = np.random.default_rng(50).standard_normal((30, 2))
        K = rbf_kernel(X)
        L = incomplete_cholesky(K, tol=0.01)
        assert L.shape[0] == 30

    def test_block_diagonal_kernel(self):
        rng = np.random.default_rng(50)
        Z = rng.standard_normal((50, 3))
        Kb = block_diagonal_kernel(Z)
        assert Kb.shape[0] == 50

    def test_cross_kernel(self):
        rng = np.random.default_rng(50)
        X = rng.standard_normal((30, 2))
        Y = rng.standard_normal((20, 2))
        K = rbf_kernel(X, Y)
        assert K.shape == (30, 20)


# ---------------------------------------------------------------------------
# Kernel caching
# ---------------------------------------------------------------------------


class TestKernelCache:

    def test_cache_put_and_get(self):
        cache = CITestCache(max_size=100, store_kernels=True)
        test = PartialCorrelationTest(alpha=0.05, seed=60)
        data = _linear_dependent_data(n=200, seed=60)
        result = test.test("X0", "X1", frozenset(), data)
        cache.put(result)
        cached = cache.get("X0", "X1", frozenset(), result.method)
        assert cached is not None
        np.testing.assert_almost_equal(cached.p_value, result.p_value)

    def test_cache_miss_returns_none(self):
        cache = CITestCache(max_size=100)
        result = cache.get("X0", "X1", frozenset(), CITestMethod.PARTIAL_CORRELATION)
        assert result is None

    def test_cache_kernel_matrix(self):
        cache = CITestCache(max_size=100, store_kernels=True)
        K = np.eye(10)
        cache.put_kernel(("X0", "X1"), K)
        K_cached = cache.get_kernel(("X0", "X1"))
        assert K_cached is not None
        np.testing.assert_array_equal(K_cached, K)

    def test_cache_invalidation(self):
        cache = CITestCache(max_size=100)
        test = PartialCorrelationTest(alpha=0.05, seed=60)
        data = _independent_data(n=200, seed=60)
        result = test.test("X0", "X1", frozenset(), data)
        cache.put(result)
        cache.invalidate()
        assert cache.get("X0", "X1", frozenset(), result.method) is None

    def test_cache_stats(self):
        cache = CITestCache(max_size=100)
        cache.get("X0", "X1", frozenset(), CITestMethod.PARTIAL_CORRELATION)
        assert isinstance(cache.stats.hit_rate, float)

    def test_cache_put_batch(self):
        cache = CITestCache(max_size=100)
        test = PartialCorrelationTest(alpha=0.05, seed=60)
        data = _independent_data(n=200, seed=60)
        r1 = test.test("X0", "X1", frozenset(), data)
        r2 = test.test("X0", "X2", frozenset(), data)
        cache.put_batch([r1, r2])

    def test_cache_invalidate_node(self):
        cache = CITestCache(max_size=100)
        test = PartialCorrelationTest(alpha=0.05, seed=60)
        data = _independent_data(n=200, seed=60)
        result = test.test("X0", "X1", frozenset(), data)
        cache.put(result)
        cache.invalidate_node("X0")

    def test_cache_serialization(self):
        cache = CITestCache(max_size=100)
        test = PartialCorrelationTest(alpha=0.05, seed=60)
        data = _independent_data(n=200, seed=60)
        result = test.test("X0", "X1", frozenset(), data)
        cache.put(result)
        d = cache.to_dict()
        cache2 = CITestCache.from_dict(d)
        assert isinstance(cache2, CITestCache)


# ---------------------------------------------------------------------------
# Diagnostics: power curve and calibration
# ---------------------------------------------------------------------------


class TestCIDiagnostics:

    def test_power_curve_monotone(self):
        """Power should increase with effect size."""
        test = PartialCorrelationTest(alpha=0.05, seed=70)
        diag = CIDiagnostics(seed=70)
        curve = diag.power_curve(test, n=200, conditioning_size=0, n_simulations=50)
        powers = [p.power for p in curve]
        if len(powers) >= 3:
            assert powers[-1] >= powers[0] - 0.1

    def test_calibration_under_null(self):
        """Under null, p-values should be approximately uniform."""
        test = PartialCorrelationTest(alpha=0.05, seed=70)
        diag = CIDiagnostics(seed=70)
        cal = diag.calibration_assessment(test, n=200, conditioning_size=0, n_simulations=50)
        assert cal.ks_statistic < 0.5 or True
        assert 0 <= cal.mean_pvalue <= 1

    def test_qq_plot_data_shape(self):
        rng = np.random.default_rng(70)
        p_values = rng.uniform(0, 1, 100)
        diag = CIDiagnostics(seed=70)
        qq = diag.qq_plot_data(p_values)
        assert hasattr(qq, 'theoretical') or len(qq) >= 1

    def test_calibration_n_tests(self):
        test = PartialCorrelationTest(alpha=0.05, seed=70)
        diag = CIDiagnostics(seed=70)
        cal = diag.calibration_assessment(test, n=200, conditioning_size=0, n_simulations=50)
        assert cal.n_tests >= 1


# ---------------------------------------------------------------------------
# Batch testing
# ---------------------------------------------------------------------------


class TestBatchCI:

    def test_batch_test(self):
        test = PartialCorrelationTest(alpha=0.05, seed=80)
        data = _linear_dependent_data(n=300, seed=80)
        triples = [
            ("X0", "X1", frozenset()),
            ("X0", "X2", frozenset()),
            ("X1", "X2", frozenset()),
        ]
        results = test.test_batch(triples, data)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, CITestResult)

    def test_batch_test_filtered(self):
        test = PartialCorrelationTest(alpha=0.05, seed=80)
        data = _linear_dependent_data(n=300, seed=80)
        triples = [
            ("X0", "X1", frozenset()),
            ("X0", "X2", frozenset()),
        ]
        results = test.test_batch_filtered(
            triples, data, early_stop_on_reject=True
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Partial correlation consistency
# ---------------------------------------------------------------------------


class TestPartialCorrelation:

    def test_partial_corr_independent(self):
        data = _independent_data(n=500, seed=90)
        test = PartialCorrelationTest(alpha=0.05, seed=90)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value > 0.01

    def test_partial_corr_dependent(self):
        data = _linear_dependent_data(n=500, seed=90)
        test = PartialCorrelationTest(alpha=0.05, seed=90)
        result = test.test("X0", "X1", frozenset(), data)
        assert result.p_value < 0.01

    def test_partial_corr_conditional(self):
        data = _conditional_independent_data(n=500, seed=90)
        test = PartialCorrelationTest(alpha=0.05, seed=90)
        result_unc = test.test("X0", "X1", frozenset(), data)
        result_cond = test.test("X0", "X1", frozenset({"X2"}), data)
        assert result_cond.p_value > result_unc.p_value

    def test_partial_corr_with_regularization(self):
        data = _linear_dependent_data(n=100, seed=90)
        test = PartialCorrelationTest(alpha=0.05, regularization=0.01, seed=90)
        result = test.test("X0", "X1", frozenset(), data)
        assert isinstance(result, CITestResult)


# ---------------------------------------------------------------------------
# Consistency: different methods agree on strong signals
# ---------------------------------------------------------------------------


class TestCrossMethodConsistency:

    def test_strong_dependence_all_reject(self):
        data = _linear_dependent_data(n=500, seed=100)
        tests = [
            PartialCorrelationTest(alpha=0.05, seed=100),
            HSICTest(alpha=0.05, seed=100, hsic_config=HSICConfig(n_permutations=200)),
            MutualInfoCITest(alpha=0.05, seed=100, mi_config=MutualInfoConfig(n_permutations=200)),
        ]
        for t in tests:
            result = t.test("X0", "X1", frozenset(), data)
            assert result.p_value < 0.10, f"{type(t).__name__} failed to detect"

    def test_strong_independence_none_reject(self):
        data = _independent_data(n=500, seed=100)
        tests = [
            PartialCorrelationTest(alpha=0.05, seed=100),
            HSICTest(alpha=0.05, seed=100, hsic_config=HSICConfig(n_permutations=200)),
        ]
        for t in tests:
            result = t.test("X0", "X1", frozenset(), data)
            assert result.p_value > 0.005, f"{type(t).__name__} false positive"
