"""Tests for parameter estimation.

Covers OLS, MLE, regularized estimation, and sufficient statistics.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.discovery.estimator import (
    ParameterEstimator,
    EstimationResult,
    SufficientStatistics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_data(rng):
    """X0 -> X1 -> X2 with known coefficients."""
    n = 500
    X0 = rng.normal(0, 1, size=n)
    X1 = 0.8 * X0 + rng.normal(0, 0.3, size=n)
    X2 = 0.6 * X1 + rng.normal(0, 0.3, size=n)
    return np.column_stack([X0, X1, X2])


@pytest.fixture
def chain_adj():
    return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


@pytest.fixture
def fork_data(rng):
    """X0 -> X1, X0 -> X2."""
    n = 500
    X0 = rng.normal(0, 1, size=n)
    X1 = 1.2 * X0 + rng.normal(0, 0.3, size=n)
    X2 = -0.7 * X0 + rng.normal(0, 0.4, size=n)
    return np.column_stack([X0, X1, X2])


@pytest.fixture
def fork_adj():
    return np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)


@pytest.fixture
def five_var_data(rng):
    n = 500
    X0 = rng.normal(0, 1, size=n)
    X1 = 0.5 * X0 + rng.normal(0, 0.5, size=n)
    X2 = 0.3 * X0 + rng.normal(0, 0.5, size=n)
    X3 = 0.4 * X1 + 0.6 * X2 + rng.normal(0, 0.3, size=n)
    X4 = 0.7 * X3 + rng.normal(0, 0.4, size=n)
    return np.column_stack([X0, X1, X2, X3, X4])


@pytest.fixture
def five_var_adj():
    adj = np.zeros((5, 5))
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    return adj


# ---------------------------------------------------------------------------
# Test OLS estimation recovery
# ---------------------------------------------------------------------------

class TestOLSEstimation:

    def test_estimate_single(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert isinstance(result, EstimationResult)
        # Should recover ~ 0.8
        assert_allclose(result.parent_coefficients[0], 0.8, atol=0.15)

    def test_estimate_all(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        results = est.estimate_all(chain_adj, linear_data)
        assert len(results) == 3

    def test_root_node_no_parents(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=0, parents=[], data=linear_data)
        assert len(result.parent_coefficients) == 0

    def test_r_squared(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert 0.0 <= result.r_squared <= 1.0

    def test_residual_variance(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert result.residual_variance > 0

    def test_t_statistics(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert len(result.t_statistics) > 0

    def test_p_values(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        for p in result.p_values:
            assert 0.0 <= p <= 1.0

    def test_is_significant(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        sig = result.is_significant(alpha=0.05)
        assert isinstance(sig, (bool, np.bool_, np.ndarray))
        if isinstance(sig, np.ndarray):
            assert sig.dtype == bool or sig.dtype == np.bool_

    def test_residuals(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert result.residuals.shape == (linear_data.shape[0],)

    def test_intercept(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert isinstance(result.intercept, (int, float, np.floating))

    def test_fork_recovery(self, fork_data, fork_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=fork_data)
        assert_allclose(result.parent_coefficients[0], 1.2, atol=0.15)

    def test_five_var(self, five_var_data, five_var_adj):
        est = ParameterEstimator(method="ols")
        results = est.estimate_all(five_var_adj, five_var_data)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Test MLE estimation
# ---------------------------------------------------------------------------

class TestMLEEstimation:

    def test_mle_single(self, linear_data, chain_adj):
        est = ParameterEstimator(method="mle")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert isinstance(result, EstimationResult)
        assert_allclose(result.parent_coefficients[0], 0.8, atol=0.15)

    def test_mle_all(self, linear_data, chain_adj):
        est = ParameterEstimator(method="mle")
        results = est.estimate_all(chain_adj, linear_data)
        assert len(results) == 3

    def test_mle_method_field(self, linear_data):
        est = ParameterEstimator(method="mle")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert result.method == "mle"


# ---------------------------------------------------------------------------
# Test regularized estimation
# ---------------------------------------------------------------------------

class TestRegularizedEstimation:

    def test_ridge(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ridge", regularization_strength=0.1)
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert isinstance(result, EstimationResult)
        # Ridge should shrink coefficients slightly
        assert abs(result.parent_coefficients[0]) < 2.0

    def test_lasso(self, linear_data, chain_adj):
        est = ParameterEstimator(method="lasso", regularization_strength=0.01)
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert isinstance(result, EstimationResult)

    def test_ridge_all(self, five_var_data, five_var_adj):
        est = ParameterEstimator(method="ridge", regularization_strength=0.1)
        results = est.estimate_all(five_var_adj, five_var_data)
        assert len(results) == 5

    def test_lasso_sparsity(self, rng):
        """Lasso should zero out irrelevant coefficients."""
        n = 500
        X = rng.normal(0, 1, size=(n, 5))
        true_coefs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Only first is relevant
        y = X @ true_coefs + rng.normal(0, 0.3, size=n)
        data = np.column_stack([X, y])
        est = ParameterEstimator(method="lasso", regularization_strength=0.1)
        result = est.estimate_single(target=5, parents=[0, 1, 2, 3, 4], data=data)
        # Irrelevant coefficients should be near zero
        assert abs(result.parent_coefficients[0]) > abs(result.parent_coefficients[2])

    def test_bayesian_estimation(self, linear_data, chain_adj):
        est = ParameterEstimator(method="bayesian")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        assert isinstance(result, EstimationResult)


# ---------------------------------------------------------------------------
# Test sufficient statistics
# ---------------------------------------------------------------------------

class TestSufficientStatistics:

    def test_from_data(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert stats.n_samples == 500
        assert stats.n_variables == 3

    def test_means_shape(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert stats.means.shape == (3,)

    def test_covariance_shape(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert stats.covariance.shape == (3, 3)

    def test_covariance_symmetric(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert_allclose(stats.covariance, stats.covariance.T, atol=1e-10)

    def test_correlation_shape(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert stats.correlation.shape == (3, 3)

    def test_correlation_diagonal_ones(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert_allclose(np.diag(stats.correlation), 1.0, atol=1e-10)

    def test_correlation_bounded(self, linear_data):
        stats = SufficientStatistics.from_data(linear_data)
        assert np.all(stats.correlation >= -1.0 - 1e-10)
        assert np.all(stats.correlation <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Test weight matrix and noise estimation
# ---------------------------------------------------------------------------

class TestWeightAndNoiseEstimation:

    def test_estimate_weight_matrix(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        W = est.estimate_weight_matrix(chain_adj, linear_data)
        assert W.shape == (3, 3)
        # Non-edge entries should be zero
        assert_allclose(W[1, 0], 0.0, atol=1e-10)

    def test_estimate_noise_variances(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        variances = est.estimate_noise_variances(chain_adj, linear_data)
        assert variances.shape == (3,)
        assert np.all(variances > 0)

    def test_compute_sufficient_stats(self, linear_data):
        est = ParameterEstimator(method="ols")
        stats = est.compute_sufficient_stats(linear_data)
        assert isinstance(stats, SufficientStatistics)


# ---------------------------------------------------------------------------
# Test residual diagnostics
# ---------------------------------------------------------------------------

class TestResidualDiagnostics:

    def test_diagnostics(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        diag = est.residual_diagnostics(result, linear_data)
        assert isinstance(diag, dict)

    def test_diagnostics_has_normality_test(self, linear_data, chain_adj):
        est = ParameterEstimator(method="ols")
        result = est.estimate_single(target=1, parents=[0], data=linear_data)
        diag = est.residual_diagnostics(result, linear_data)
        # Should have some normality test result
        assert any(k in diag for k in ["shapiro", "normality", "dagostino", "shapiro_wilk"])


# ---------------------------------------------------------------------------
# Test EstimationResult dataclass
# ---------------------------------------------------------------------------

class TestEstimationResult:

    def test_n_params(self):
        result = EstimationResult(
            target=1, parents=[0, 2],
            coefficients=np.array([0.1, 0.8, 0.3]),
            standard_errors=np.array([0.05, 0.1, 0.1]),
            t_statistics=np.array([2.0, 8.0, 3.0]),
            p_values=np.array([0.04, 0.001, 0.01]),
            residual_variance=0.5,
            r_squared=0.85,
            adj_r_squared=0.84,
            residuals=np.zeros(100),
            method="ols",
        )
        assert result.n_params == 3

    def test_intercept_property(self):
        result = EstimationResult(
            target=1, parents=[0],
            coefficients=np.array([0.5, 0.8]),
            standard_errors=np.array([0.1, 0.1]),
            t_statistics=np.array([5.0, 8.0]),
            p_values=np.array([0.001, 0.001]),
            residual_variance=0.3,
            r_squared=0.9,
            adj_r_squared=0.89,
            residuals=np.zeros(100),
            method="ols",
        )
        assert_allclose(result.intercept, 0.5)

    def test_parent_coefficients(self):
        result = EstimationResult(
            target=1, parents=[0],
            coefficients=np.array([0.5, 0.8]),
            standard_errors=np.array([0.1, 0.1]),
            t_statistics=np.array([5.0, 8.0]),
            p_values=np.array([0.001, 0.001]),
            residual_variance=0.3,
            r_squared=0.9,
            adj_r_squared=0.89,
            residuals=np.zeros(100),
            method="ols",
        )
        assert_allclose(result.parent_coefficients, [0.8])

    def test_to_dict(self):
        result = EstimationResult(
            target=1, parents=[0],
            coefficients=np.array([0.5, 0.8]),
            standard_errors=np.array([0.1, 0.1]),
            t_statistics=np.array([5.0, 8.0]),
            p_values=np.array([0.001, 0.001]),
            residual_variance=0.3,
            r_squared=0.9,
            adj_r_squared=0.89,
            residuals=np.zeros(100),
            method="ols",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "target" in d
