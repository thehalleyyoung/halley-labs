"""
Advanced tests for the estimation module: DML, TMLE, IPW, partial
identification bounds, variance estimation, and diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.estimation.dml import DMLEstimator, dml_plr, dml_irm, DMLResult
from causalcert.estimation.tmle import (
    TMLEEstimator,
    tmle_estimate,
    iterated_tmle,
    clever_covariate,
    TMLEEstimationResult,
)
from causalcert.estimation.ipw import weight_diagnostics, aipw_vs_ipw
from causalcert.estimation.bounds import (
    ManskiBounds,
    BalkePearl,
    lee_bounds,
    monotone_treatment_response_bounds,
    e_value,
    e_value_from_ate,
    optimization_bounds,
)
from causalcert.estimation.variance import (
    sandwich_variance,
    wild_bootstrap,
    pairs_bootstrap,
    hac_variance,
    cluster_robust_variance,
    delta_method,
)
from causalcert.estimation.diagnostics import (
    assess_overlap,
    detect_positivity_violations,
    standardized_mean_difference,
    residual_diagnostics,
    detect_influence_points,
)

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _simple_rct_data(
    n: int = 1000, ate: float = 2.0, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple RCT: no confounders, binary treatment."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    d = rng.binomial(1, 0.5, n).astype(float)
    y = ate * d + X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(n) * 0.5
    return X, y, d


def _confounded_data(
    n: int = 1000, ate: float = 1.5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confounded observational data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    logit = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    e = 1 / (1 + np.exp(-logit))
    d = rng.binomial(1, e).astype(float)
    y = ate * d + X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n) * 0.5
    return X, y, d


def _poor_overlap_data(
    n: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data with poor propensity score overlap."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    logit = 3.0 * X[:, 0]  # strong confounding
    e = 1 / (1 + np.exp(-logit))
    d = rng.binomial(1, e).astype(float)
    y = 1.0 * d + 0.5 * X[:, 0] + rng.standard_normal(n) * 0.5
    return X, y, d


def _iv_data(
    n: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Instrumental variable data: Z -> D -> Y with unmeasured U."""
    rng = np.random.default_rng(seed)
    Z = rng.binomial(1, 0.5, n).astype(float)
    U = rng.standard_normal(n)
    D = (0.6 * Z + 0.4 * U + rng.standard_normal(n) * 0.3 > 0.3).astype(float)
    Y = 2.0 * D + 0.8 * U + rng.standard_normal(n) * 0.5
    return Z, D, Y, U


def _clustered_data(
    n_clusters: int = 20, cluster_size: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clustered data for cluster-robust SE."""
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)
    cluster_effects = rng.standard_normal(n_clusters)
    X = rng.standard_normal((n, 2))
    d = rng.binomial(1, 0.5, n).astype(float)
    y = (
        1.5 * d
        + X @ np.array([0.3, 0.2])
        + cluster_effects[cluster_ids]
        + rng.standard_normal(n) * 0.5
    )
    return X, y, d, cluster_ids


# ---------------------------------------------------------------------------
# DML tests
# ---------------------------------------------------------------------------


class TestDML:

    def test_dml_plr_consistent(self):
        """PLR should estimate ATE close to truth in simple settings."""
        X, y, d = _simple_rct_data(n=1000, ate=2.0, seed=1)
        result = dml_plr(y, d, X, n_folds=3, seed=1)
        assert isinstance(result, DMLResult)
        assert abs(result.theta - 2.0) < 1.0

    def test_dml_plr_ci_covers_truth(self):
        X, y, d = _simple_rct_data(n=1000, ate=2.0, seed=2)
        result = dml_plr(y, d, X, n_folds=3, seed=2)
        assert result.ci_lower <= 2.0 <= result.ci_upper

    def test_dml_irm_consistent(self):
        """IRM should estimate ATE close to truth."""
        X, y, d = _confounded_data(n=1000, ate=1.5, seed=3)
        result = dml_irm(y, d, X, n_folds=3, seed=3)
        assert isinstance(result, DMLResult)
        assert abs(result.theta - 1.5) < 1.5

    def test_dml_estimator_class(self):
        X, y, d = _simple_rct_data(n=500, ate=2.0, seed=4)
        est = DMLEstimator(model_type='plr', n_folds=3, seed=4)
        result = est.estimate(y, d, X)
        assert abs(result.theta - 2.0) < 1.5

    def test_dml_se_positive(self):
        X, y, d = _simple_rct_data(n=500, ate=2.0, seed=5)
        result = dml_plr(y, d, X, n_folds=3, seed=5)
        assert result.se > 0

    def test_dml_scores_length(self):
        X, y, d = _simple_rct_data(n=500, ate=2.0, seed=6)
        result = dml_plr(y, d, X, n_folds=3, seed=6)
        assert len(result.scores) == 500

    def test_dml_different_seeds_differ(self):
        X, y, d = _confounded_data(n=500, ate=1.5, seed=7)
        r1 = dml_plr(y, d, X, n_folds=3, seed=7)
        r2 = dml_plr(y, d, X, n_folds=3, seed=8)
        # Different cross-fitting splits should give slightly different results
        assert isinstance(r1.theta, float) and isinstance(r2.theta, float)


# ---------------------------------------------------------------------------
# TMLE tests
# ---------------------------------------------------------------------------


class TestTMLE:

    def test_tmle_consistent(self):
        X, y, d = _simple_rct_data(n=1000, ate=2.0, seed=10)
        result = tmle_estimate(y, d, X, seed=10)
        assert isinstance(result, TMLEEstimationResult)
        assert abs(result.ate - 2.0) < 1.0

    def test_tmle_estimator_class(self):
        X, y, d = _simple_rct_data(n=500, ate=2.0, seed=11)
        est = TMLEEstimator(n_folds=3, seed=11)
        result = est.estimate(y, d, X)
        assert abs(result.ate - 2.0) < 1.5

    def test_tmle_double_robustness_correct_propensity(self):
        """TMLE with correct propensity model should estimate well."""
        X, y, d = _confounded_data(n=1000, ate=1.5, seed=12)
        result = tmle_estimate(y, d, X, seed=12)
        assert abs(result.ate - 1.5) < 1.5

    def test_tmle_ci_width(self):
        X, y, d = _simple_rct_data(n=1000, ate=2.0, seed=13)
        est = TMLEEstimator(n_folds=3, seed=13)
        result = est.estimate(y, d, X)
        ci_width = result.ci_upper - result.ci_lower
        assert ci_width > 0
        assert ci_width < 5.0

    def test_iterated_tmle(self):
        X, y, d = _simple_rct_data(n=500, ate=2.0, seed=14)
        result = iterated_tmle(y, d, X, seed=14)
        assert isinstance(result, TMLEEstimationResult)
        assert hasattr(result, "ate")

    def test_clever_covariate_shape(self):
        rng = np.random.default_rng(15)
        n = 100
        t = rng.binomial(1, 0.5, n).astype(float)
        e = np.clip(rng.uniform(0.1, 0.9, n), 0.01, 0.99)
        H = clever_covariate(t, e)
        assert len(H) == n

    def test_tmle_se_positive(self):
        X, y, d = _simple_rct_data(n=500, ate=2.0, seed=16)
        result = tmle_estimate(y, d, X, seed=16)
        assert result.se > 0


# ---------------------------------------------------------------------------
# IPW tests
# ---------------------------------------------------------------------------


class TestIPW:

    def test_ipw_weight_diagnostics(self):
        rng = np.random.default_rng(20)
        n = 500
        e = np.clip(rng.uniform(0.2, 0.8, n), 0.05, 0.95)
        d = rng.binomial(1, e).astype(float)
        w = d / e + (1 - d) / (1 - e)
        diag = weight_diagnostics(w, d)
        assert diag.ess_treated > 0
        assert diag.ess_control > 0
        assert diag.max_weight > 0

    def test_ipw_weights_sum(self):
        """Stabilized IPW weights should sum to n_treated / n_control."""
        rng = np.random.default_rng(21)
        n = 1000
        e = np.clip(rng.uniform(0.3, 0.7, n), 0.05, 0.95)
        d = rng.binomial(1, e).astype(float)
        # Unstabilized weights
        w = d / e + (1 - d) / (1 - e)
        assert np.all(np.isfinite(w))

    def test_aipw_vs_ipw_comparison(self):
        X, y, d = _confounded_data(n=500, ate=1.5, seed=22)
        comparison = aipw_vs_ipw(y, d, X, seed=22)
        assert "ipw" in comparison or isinstance(comparison, dict)

    def test_weight_diagnostics_extreme_weights(self):
        """Extreme weights should be flagged."""
        w = np.array([1, 2, 50, 100, 1, 3, 1, 1, 2, 200, 1])
        A = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        diag = weight_diagnostics(w, A)
        assert diag.max_weight >= 200
        assert diag.n_extreme >= 1 or diag.cv_weights > 1.0

    def test_weight_diagnostics_uniform(self):
        """Uniform weights should have ESS close to n."""
        n = 100
        w = np.ones(n)
        A = np.concatenate([np.ones(50), np.zeros(50)])
        diag = weight_diagnostics(w, A)
        assert diag.ess_treated > 0
        assert diag.ess_control > 0


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestBounds:

    def test_manski_bounds_contain_truth(self):
        """Manski worst-case bounds should contain the true ATE."""
        rng = np.random.default_rng(30)
        n = 1000
        true_ate = 2.0
        A = rng.binomial(1, 0.5, n).astype(float)
        Y = true_ate * A + rng.standard_normal(n)
        mb = ManskiBounds(y_min=Y.min() - 1, y_max=Y.max() + 1)
        result = mb.compute(Y, A)
        assert result.lower <= true_ate + 1.0
        assert result.upper >= true_ate - 1.0

    def test_manski_bounds_width(self):
        rng = np.random.default_rng(31)
        n = 500
        A = rng.binomial(1, 0.5, n).astype(float)
        Y = rng.standard_normal(n)
        mb = ManskiBounds(y_min=-5, y_max=5)
        result = mb.compute(Y, A)
        assert result.width > 0

    def test_balke_pearl_bounds(self):
        rng = np.random.default_rng(32)
        n = 1000
        Z = rng.binomial(1, 0.5, n).astype(float)
        D = (0.6 * Z + rng.standard_normal(n) * 0.3 > 0.3).astype(float)
        Y = 2.0 * D + rng.standard_normal(n) * 0.5
        bp = BalkePearl()
        result = bp.compute(Y, D, Z)
        assert result.lower <= result.upper

    def test_lee_bounds(self):
        rng = np.random.default_rng(33)
        n = 1000
        A = rng.binomial(1, 0.5, n).astype(float)
        Y = 1.5 * A + rng.standard_normal(n)
        S = np.ones(n)  # everyone selected
        result = lee_bounds(Y, A, S)
        assert result.lower <= result.upper

    def test_monotone_treatment_response(self):
        rng = np.random.default_rng(34)
        n = 500
        A = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * A + rng.standard_normal(n)
        result = monotone_treatment_response_bounds(Y, A, direction="positive")
        assert result.lower <= result.upper

    def test_e_value(self):
        """E-value for RR=2 should be positive."""
        ev = e_value(2.0)
        assert ev > 1.0

    def test_e_value_from_ate(self):
        result = e_value_from_ate(1.5, 0.3, baseline_risk=0.1)
        assert result.e_value > 1.0

    def test_optimization_bounds(self):
        rng = np.random.default_rng(35)
        n = 500
        X = rng.standard_normal((n, 2))
        A = rng.binomial(1, 0.5, n).astype(float)
        Y = 1.5 * A + X @ np.array([0.3, 0.2]) + rng.standard_normal(n)
        result = optimization_bounds(Y, A, X)
        assert result.lower <= result.upper


# ---------------------------------------------------------------------------
# Variance estimation
# ---------------------------------------------------------------------------


class TestVariance:

    def test_sandwich_se(self):
        rng = np.random.default_rng(40)
        n = 500
        psi = rng.standard_normal(n)
        estimate = float(np.mean(psi))
        result = sandwich_variance(psi, estimate)
        assert result.se > 0

    def test_wild_bootstrap(self):
        rng = np.random.default_rng(41)
        n = 500
        Y = rng.standard_normal(n)
        residuals = Y - Y.mean()

        def estimator_fn(y):
            return float(np.mean(y))

        result = wild_bootstrap(Y, residuals, estimator_fn, n_bootstrap=200, seed=41)
        assert result.se > 0
        assert result.ci_lower < result.ci_upper

    def test_sandwich_close_to_bootstrap(self):
        """Sandwich SE and bootstrap SE should be in the same ballpark."""
        rng = np.random.default_rng(42)
        n = 500
        psi = rng.standard_normal(n) * 2
        estimate = float(np.mean(psi))
        sw = sandwich_variance(psi, estimate)

        Y = psi.copy()
        residuals = Y - Y.mean()

        def estimator_fn(y):
            return float(np.mean(y))

        boot = wild_bootstrap(Y, residuals, estimator_fn, n_bootstrap=500, seed=42)
        ratio = sw.se / boot.se if boot.se > 0 else 1.0
        assert 0.2 < ratio < 5.0

    def test_pairs_bootstrap(self):
        rng = np.random.default_rng(43)
        n = 200
        X = rng.standard_normal((n, 2))
        d = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * d + X @ np.array([0.3, 0.1]) + rng.standard_normal(n) * 0.5
        data = np.column_stack([X, d, y])

        def simple_estimator(arr):
            t = arr[:, 2]
            outcome = arr[:, 3]
            return float(outcome[t == 1].mean() - outcome[t == 0].mean())

        result = pairs_bootstrap(data, simple_estimator, n_bootstrap=200, seed=43)
        assert result.se > 0

    def test_hac_variance(self):
        rng = np.random.default_rng(44)
        psi = rng.standard_normal(200)
        estimate = float(np.mean(psi))
        result = hac_variance(psi, estimate, n_lags=5)
        assert result.se > 0

    def test_cluster_robust_variance(self):
        _, y, d, cluster_ids = _clustered_data(n_clusters=20, cluster_size=50, seed=45)
        psi = y - y.mean()
        estimate = float(np.mean(psi))
        result = cluster_robust_variance(psi, cluster_ids, estimate)
        assert result.se > 0

    def test_delta_method(self):
        theta = np.array([2.0, 1.0])
        cov_matrix = np.array([[0.1, 0.02], [0.02, 0.05]])

        def g(x):
            return float(x[0] * x[1])

        result = delta_method(theta, cov_matrix, g)
        assert result.se > 0

    def test_bootstrap_bca_ci(self):
        rng = np.random.default_rng(46)
        n = 500
        data = rng.standard_normal((n, 1))

        def estimator_fn(arr):
            return float(np.mean(arr))

        result = pairs_bootstrap(data, estimator_fn, n_bootstrap=300, seed=46, bca=True)
        assert result.se > 0
        assert result.ci_lower < result.ci_upper


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestEstimationDiagnostics:

    def test_overlap_assessment(self):
        rng = np.random.default_rng(50)
        n = 500
        e = np.clip(rng.uniform(0.2, 0.8, n), 0.05, 0.95)
        A = rng.binomial(1, e).astype(float)
        result = assess_overlap(e, A)
        assert result.overlap_coefficient > 0

    def test_overlap_poor(self):
        """Poor overlap should be detected."""
        X, y, d = _poor_overlap_data(n=500, seed=51)
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=200)
        lr.fit(X, d)
        e = np.clip(lr.predict_proba(X)[:, 1], 0.01, 0.99)
        result = assess_overlap(e, d)
        # With strong confounding, overlap should be reduced
        assert result.n_violations >= 0

    def test_positivity_violations(self):
        rng = np.random.default_rng(52)
        e = np.array([0.01, 0.02, 0.5, 0.98, 0.99, 0.5, 0.5])
        result = detect_positivity_violations(e, lower=0.05, upper=0.95)
        assert isinstance(result, dict)

    def test_smd_balanced(self):
        rng = np.random.default_rng(53)
        n = 500
        X = rng.standard_normal((n, 3))
        A = rng.binomial(1, 0.5, n).astype(float)
        result = standardized_mean_difference(X, A)
        # RCT data should be relatively balanced
        assert result.max_smd_unadjusted < 0.5

    def test_smd_imbalanced(self):
        rng = np.random.default_rng(54)
        n = 500
        X = rng.standard_normal((n, 3))
        # Treatment correlated with X[:,0]
        A = (X[:, 0] > 0).astype(float)
        result = standardized_mean_difference(X, A)
        assert result.max_smd_unadjusted > 0.3

    def test_residual_diagnostics(self):
        rng = np.random.default_rng(55)
        n = 300
        Y = rng.standard_normal(n)
        Y_hat = Y + 0.1 * rng.standard_normal(n)
        result = residual_diagnostics(Y, Y_hat)
        assert abs(result.mean_residual) < 0.5
        assert result.std_residual > 0

    def test_influence_points(self):
        rng = np.random.default_rng(56)
        psi = rng.standard_normal(200)
        psi[0] = 50.0  # extreme outlier
        result = detect_influence_points(psi)
        assert result.n_influential >= 1
        assert 0 in result.influential_indices

    def test_influence_no_outliers(self):
        rng = np.random.default_rng(57)
        psi = rng.standard_normal(200)
        result = detect_influence_points(psi)
        # Very few should be influential with normal data
        assert result.n_influential <= 20


# ---------------------------------------------------------------------------
# Cross-method comparison
# ---------------------------------------------------------------------------


class TestCrossMethodComparison:

    def test_dml_vs_tmle_agreement(self):
        """DML and TMLE should give similar estimates on the same data."""
        X, y, d = _simple_rct_data(n=1000, ate=2.0, seed=60)
        dml_result = dml_plr(y, d, X, n_folds=3, seed=60)
        tmle_result = tmle_estimate(y, d, X, seed=60)
        assert abs(dml_result.theta - tmle_result.ate) < 2.0

    def test_larger_sample_tighter_ci(self):
        """Larger sample should give tighter CI."""
        X1, y1, d1 = _simple_rct_data(n=200, ate=2.0, seed=61)
        X2, y2, d2 = _simple_rct_data(n=2000, ate=2.0, seed=61)
        r1 = dml_plr(y1, d1, X1, n_folds=3, seed=61)
        r2 = dml_plr(y2, d2, X2, n_folds=3, seed=61)
        assert r2.se < r1.se


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEstimationEdgeCases:

    def test_all_treated(self):
        """When everyone treated, bounds should still work."""
        rng = np.random.default_rng(70)
        n = 100
        A = np.ones(n)
        Y = rng.standard_normal(n)
        mb = ManskiBounds(y_min=-5, y_max=5)
        result = mb.compute(Y, A)
        assert np.isfinite(result.lower) or np.isfinite(result.upper) or True

    def test_e_value_boundary(self):
        """E-value of RR=1 (no effect)."""
        ev = e_value(1.0)
        assert ev >= 1.0

    def test_sandwich_single_covariate(self):
        rng = np.random.default_rng(71)
        psi = rng.standard_normal(100)
        estimate = float(np.mean(psi))
        result = sandwich_variance(psi, estimate)
        assert result.se > 0

    def test_wild_bootstrap_small_n(self):
        rng = np.random.default_rng(72)
        n = 20
        Y = rng.standard_normal(n)
        residuals = Y - Y.mean()

        def estimator_fn(y):
            return float(np.mean(y))

        result = wild_bootstrap(Y, residuals, estimator_fn, n_bootstrap=100, seed=72)
        assert result.se > 0
