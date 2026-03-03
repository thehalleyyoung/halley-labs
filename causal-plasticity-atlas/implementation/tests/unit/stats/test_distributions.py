"""Tests for cpa.stats.distributions module."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats as sp_stats

from cpa.stats.distributions import (
    GaussianConditional,
    bh_fdr_correction,
    bonferroni_correction,
    bootstrap_ci,
    fisher_z_test,
    jsd_discrete,
    jsd_gaussian,
    kl_discrete,
    kl_gaussian,
    kl_gaussian_mv,
    partial_correlation,
    partial_correlation_matrix,
    partial_correlation_test,
    sqrt_jsd_discrete,
    sqrt_jsd_gaussian,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_gc():
    """Y = 0.5*X1 - 0.3*X2 + 1.0 + eps, eps ~ N(0, 0.25)."""
    return GaussianConditional(
        "Y", ["X1", "X2"], np.array([0.5, -0.3]),
        intercept=1.0, residual_variance=0.25,
    )


@pytest.fixture
def no_parent_gc():
    """Y ~ N(2.0, 1.0) with no parents."""
    return GaussianConditional(
        "Y", [], np.array([]), intercept=2.0, residual_variance=1.0,
    )


# ===================================================================
# GaussianConditional — construction & validation
# ===================================================================


class TestGaussianConditionalConstruction:

    def test_basic_creation(self, simple_gc):
        assert simple_gc.variable == "Y"
        assert simple_gc.parents == ["X1", "X2"]
        assert simple_gc.num_parents == 2
        np.testing.assert_array_equal(simple_gc.coefficients, [0.5, -0.3])
        assert simple_gc.intercept == 1.0
        assert simple_gc.residual_variance == 0.25

    def test_residual_std(self, simple_gc):
        assert simple_gc.residual_std == pytest.approx(math.sqrt(0.25))

    def test_no_parents(self, no_parent_gc):
        assert no_parent_gc.num_parents == 0
        assert no_parent_gc.residual_std == pytest.approx(1.0)

    def test_coefficients_coerced_to_float64(self):
        gc = GaussianConditional("Y", ["X"], np.array([1], dtype=np.int32),
                                 residual_variance=1.0)
        assert gc.coefficients.dtype == np.float64

    def test_invalid_2d_coefficients(self):
        with pytest.raises(ValueError, match="1-D"):
            GaussianConditional("Y", ["X"], np.array([[1.0, 2.0]]),
                                residual_variance=1.0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            GaussianConditional("Y", ["X1", "X2"], np.array([1.0]),
                                residual_variance=1.0)

    @pytest.mark.parametrize("var", [0.0, -1.0, -1e-10])
    def test_non_positive_variance(self, var):
        with pytest.raises(ValueError, match="residual_variance"):
            GaussianConditional("Y", [], np.array([]),
                                residual_variance=var)

    def test_repr_contains_variable(self, simple_gc):
        r = repr(simple_gc)
        assert "Y" in r
        assert "GaussianConditional" in r


# ===================================================================
# GaussianConditional — mean / log_prob / sample
# ===================================================================


class TestGaussianConditionalInference:

    def test_mean_computation(self, simple_gc):
        # 0.5*2 - 0.3*1 + 1.0 = 1.7
        mu = simple_gc.mean(np.array([2.0, 1.0]))
        assert mu == pytest.approx(1.7)

    def test_mean_no_parents(self, no_parent_gc):
        mu = no_parent_gc.mean(np.array([]))
        assert mu == pytest.approx(2.0)

    def test_mean_wrong_shape_raises(self, simple_gc):
        with pytest.raises(ValueError, match="parent_values shape"):
            simple_gc.mean(np.array([1.0]))

    def test_log_prob_matches_scipy(self, simple_gc):
        pv = np.array([2.0, 1.0])
        mu = simple_gc.mean(pv)
        expected = sp_stats.norm.logpdf(3.0, loc=mu, scale=simple_gc.residual_std)
        assert simple_gc.log_prob(3.0, pv) == pytest.approx(expected)

    def test_log_prob_at_mean_is_max(self, simple_gc):
        pv = np.array([0.0, 0.0])
        mu = simple_gc.mean(pv)
        lp_at_mean = simple_gc.log_prob(mu, pv)
        lp_offset = simple_gc.log_prob(mu + 2.0, pv)
        assert lp_at_mean > lp_offset

    def test_sample_shape_1d(self, simple_gc, rng):
        samples = simple_gc.sample(np.array([1.0, 0.0]), n=500, rng=rng)
        assert samples.shape == (500,)

    def test_sample_mean_close(self, simple_gc, rng):
        pv = np.array([2.0, 1.0])
        samples = simple_gc.sample(pv, n=10_000, rng=rng)
        assert np.mean(samples) == pytest.approx(1.7, abs=0.05)

    def test_sample_std_close(self, simple_gc, rng):
        pv = np.array([2.0, 1.0])
        samples = simple_gc.sample(pv, n=10_000, rng=rng)
        assert np.std(samples) == pytest.approx(0.5, abs=0.05)

    def test_sample_2d_parent_values(self, simple_gc, rng):
        pv = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        samples = simple_gc.sample(pv, rng=rng)
        assert samples.shape == (3,)

    def test_sample_invalid_ndim_raises(self, simple_gc, rng):
        with pytest.raises(ValueError):
            simple_gc.sample(np.ones((2, 2, 2)), rng=rng)


# ===================================================================
# GaussianConditional — KL / JSD
# ===================================================================


class TestGaussianConditionalDivergences:

    def test_kl_identical_is_zero(self):
        gc = GaussianConditional("Y", ["X"], np.array([1.0]),
                                 intercept=0.0, residual_variance=1.0)
        assert gc.kl_divergence_to(gc) == pytest.approx(0.0, abs=1e-12)

    def test_kl_different_intercept(self):
        gc1 = GaussianConditional("Y", [], np.array([]),
                                  intercept=0.0, residual_variance=1.0)
        gc2 = GaussianConditional("Y", [], np.array([]),
                                  intercept=1.0, residual_variance=1.0)
        expected = kl_gaussian(0.0, 1.0, 1.0, 1.0)
        assert gc1.kl_divergence_to(gc2) == pytest.approx(expected)

    def test_kl_nonnegative(self):
        gc1 = GaussianConditional("Y", [], np.array([]),
                                  intercept=0.0, residual_variance=1.0)
        gc2 = GaussianConditional("Y", [], np.array([]),
                                  intercept=3.0, residual_variance=2.0)
        assert gc1.kl_divergence_to(gc2) >= 0.0

    def test_jsd_identical_is_zero(self):
        gc = GaussianConditional("Y", [], np.array([]),
                                 intercept=0.0, residual_variance=1.0)
        assert gc.jsd_to(gc) == pytest.approx(0.0, abs=1e-8)

    def test_jsd_symmetric(self):
        gc1 = GaussianConditional("Y", [], np.array([]),
                                  intercept=0.0, residual_variance=1.0)
        gc2 = GaussianConditional("Y", [], np.array([]),
                                  intercept=2.0, residual_variance=0.5)
        assert gc1.jsd_to(gc2) == pytest.approx(gc2.jsd_to(gc1), abs=1e-8)


# ===================================================================
# GaussianConditional — serialization
# ===================================================================


class TestGaussianConditionalSerialization:

    def test_round_trip(self, simple_gc):
        d = simple_gc.to_dict()
        restored = GaussianConditional.from_dict(d)
        assert restored.variable == simple_gc.variable
        assert restored.parents == simple_gc.parents
        np.testing.assert_array_almost_equal(
            restored.coefficients, simple_gc.coefficients,
        )
        assert restored.intercept == simple_gc.intercept
        assert restored.residual_variance == simple_gc.residual_variance

    def test_dict_keys(self, simple_gc):
        d = simple_gc.to_dict()
        assert set(d.keys()) == {
            "variable", "parents", "coefficients",
            "intercept", "residual_variance",
        }

    def test_coefficients_serialized_as_list(self, simple_gc):
        d = simple_gc.to_dict()
        assert isinstance(d["coefficients"], list)

    def test_from_dict_defaults(self):
        d = {"variable": "Z", "parents": [], "coefficients": []}
        gc = GaussianConditional.from_dict(d)
        assert gc.intercept == 0.0
        assert gc.residual_variance == 1.0


# ===================================================================
# GaussianConditional — fit
# ===================================================================


class TestGaussianConditionalFit:

    def test_fit_recovers_parameters(self, rng):
        n = 5000
        X1 = rng.normal(0, 1, n)
        X2 = rng.normal(0, 1, n)
        Y = 2.0 * X1 - 1.0 * X2 + 3.0 + rng.normal(0, 0.5, n)
        X_parents = np.column_stack([X1, X2])
        gc = GaussianConditional.fit("Y", ["X1", "X2"], Y, X_parents)
        assert gc.coefficients[0] == pytest.approx(2.0, abs=0.1)
        assert gc.coefficients[1] == pytest.approx(-1.0, abs=0.1)
        assert gc.intercept == pytest.approx(3.0, abs=0.1)
        assert gc.residual_variance == pytest.approx(0.25, abs=0.1)

    def test_fit_no_parents(self, rng):
        Y = rng.normal(5.0, 2.0, 1000)
        gc = GaussianConditional.fit("Y", [], Y, np.empty((1000, 0)))
        assert gc.intercept == pytest.approx(5.0, abs=0.2)
        assert gc.num_parents == 0

    def test_fit_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same n"):
            GaussianConditional.fit(
                "Y", ["X"], np.ones(10), np.ones((5, 1)),
            )


# ===================================================================
# KL divergence — discrete
# ===================================================================


class TestKLDiscrete:

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_discrete(p, p) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("p,q,expected", [
        (np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.0),
        (np.array([1.0, 0.0]), np.array([0.5, 0.5]), math.log(2.0)),
    ])
    def test_known_values(self, p, q, expected):
        assert kl_discrete(p, q) == pytest.approx(expected, abs=1e-10)

    def test_disjoint_support_returns_inf(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert kl_discrete(p, q) == float("inf")

    def test_non_negative(self, rng):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        assert kl_discrete(p, q) >= -1e-12

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            kl_discrete(np.array([0.5, 0.5]), np.array([1.0 / 3] * 3))

    def test_negative_values_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            kl_discrete(np.array([-0.5, 1.5]), np.array([0.5, 0.5]))

    def test_normalization_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kl_discrete(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
            assert any("normalizing" in str(x.message).lower() for x in w)


# ===================================================================
# KL divergence — Gaussian
# ===================================================================


class TestKLGaussian:

    def test_identical_is_zero(self):
        assert kl_gaussian(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-12)

    def test_analytic_value(self):
        # KL(N(0,1) || N(1,1)) = 0.5 * (0 + 1 + 1 - 1) = 0.5
        assert kl_gaussian(0.0, 1.0, 1.0, 1.0) == pytest.approx(0.5)

    def test_different_variance(self):
        # KL(N(0,1) || N(0,2)) = 0.5*(ln(2) + 0.5 - 1) = 0.5*(ln2 - 0.5)
        expected = 0.5 * (math.log(2.0) + 0.5 - 1.0)
        assert kl_gaussian(0.0, 1.0, 0.0, 2.0) == pytest.approx(expected)

    def test_non_negative(self, rng):
        for _ in range(20):
            mu1, mu2 = rng.normal(0, 5, 2)
            var1, var2 = rng.exponential(2, 2)
            assert kl_gaussian(mu1, var1, mu2, var2) >= -1e-12

    @pytest.mark.parametrize("bad_var1,bad_var2", [
        (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0),
    ])
    def test_non_positive_variance_raises(self, bad_var1, bad_var2):
        with pytest.raises(ValueError, match="Variances must be > 0"):
            kl_gaussian(0.0, bad_var1, 0.0, bad_var2)


# ===================================================================
# KL divergence — multivariate Gaussian
# ===================================================================


class TestKLGaussianMV:

    def test_identical_is_zero(self):
        mu = np.array([0.0, 0.0])
        cov = np.eye(2)
        assert kl_gaussian_mv(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-10)

    def test_reduces_to_univariate(self):
        mu1, var1, mu2, var2 = 1.0, 2.0, 3.0, 4.0
        kl_uni = kl_gaussian(mu1, var1, mu2, var2)
        kl_mv = kl_gaussian_mv(
            np.array([mu1]), np.array([[var1]]),
            np.array([mu2]), np.array([[var2]]),
        )
        assert kl_mv == pytest.approx(kl_uni, abs=1e-10)

    def test_non_negative(self, rng):
        d = 3
        mu1, mu2 = rng.normal(0, 1, d), rng.normal(0, 1, d)
        A = rng.normal(0, 1, (d, d))
        cov1 = A @ A.T + 0.1 * np.eye(d)
        B = rng.normal(0, 1, (d, d))
        cov2 = B @ B.T + 0.1 * np.eye(d)
        assert kl_gaussian_mv(mu1, cov1, mu2, cov2) >= -1e-10


# ===================================================================
# JSD — discrete
# ===================================================================


class TestJSDDiscrete:

    def test_identical_is_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert jsd_discrete(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric(self, rng):
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        assert jsd_discrete(p, q) == pytest.approx(jsd_discrete(q, p), abs=1e-12)

    def test_bounded_by_ln2(self, rng):
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        assert 0.0 <= jsd_discrete(p, q) <= math.log(2.0) + 1e-10

    def test_disjoint_support_is_ln2(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert jsd_discrete(p, q) == pytest.approx(math.log(2.0), abs=1e-10)

    def test_sqrt_jsd_is_metric(self, rng):
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        r = rng.dirichlet(np.ones(4))
        d_pq = sqrt_jsd_discrete(p, q)
        d_qr = sqrt_jsd_discrete(q, r)
        d_pr = sqrt_jsd_discrete(p, r)
        assert d_pr <= d_pq + d_qr + 1e-10  # triangle inequality


# ===================================================================
# JSD — Gaussian
# ===================================================================


class TestJSDGaussian:

    def test_identical_is_zero(self):
        assert jsd_gaussian(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-8)

    def test_symmetric(self):
        j1 = jsd_gaussian(0.0, 1.0, 2.0, 3.0)
        j2 = jsd_gaussian(2.0, 3.0, 0.0, 1.0)
        assert j1 == pytest.approx(j2, abs=1e-8)

    def test_non_negative(self, rng):
        for _ in range(10):
            mu1, mu2 = rng.normal(0, 3, 2)
            var1, var2 = rng.exponential(2, 2)
            assert jsd_gaussian(mu1, var1, mu2, var2) >= -1e-10

    def test_bounded_by_ln2(self):
        # Far apart Gaussians → JSD approaches ln(2)
        j = jsd_gaussian(0.0, 0.01, 100.0, 0.01)
        assert j <= math.log(2.0) + 1e-6

    @pytest.mark.parametrize("bad_var1,bad_var2", [(0.0, 1.0), (1.0, -1.0)])
    def test_non_positive_variance_raises(self, bad_var1, bad_var2):
        with pytest.raises(ValueError, match="Variances must be > 0"):
            jsd_gaussian(0.0, bad_var1, 0.0, bad_var2)

    def test_sqrt_jsd(self):
        sj = sqrt_jsd_gaussian(0.0, 1.0, 1.0, 1.0)
        j = jsd_gaussian(0.0, 1.0, 1.0, 1.0)
        assert sj == pytest.approx(math.sqrt(j), abs=1e-8)


# ===================================================================
# Partial correlation
# ===================================================================


class TestPartialCorrelation:

    def test_uncorrelated_variables(self, rng):
        n = 2000
        X = rng.normal(0, 1, (n, 3))  # independent columns
        r = partial_correlation(X, 0, 1)
        assert abs(r) < 0.1

    def test_correlated_via_confounder(self, rng):
        n = 2000
        Z = rng.normal(0, 1, n)
        X0 = Z + 0.1 * rng.normal(0, 1, n)
        X1 = Z + 0.1 * rng.normal(0, 1, n)
        X = np.column_stack([X0, X1, Z])
        # Marginal correlation should be high
        r_marginal = partial_correlation(X, 0, 1)
        assert abs(r_marginal) > 0.5
        # Partial correlation conditioning on Z should be near 0
        r_partial = partial_correlation(X, 0, 1, [2])
        assert abs(r_partial) < 0.15

    def test_too_few_samples_raises(self):
        X = np.ones((2, 3))
        with pytest.raises(ValueError, match="at least 3"):
            partial_correlation(X, 0, 1)

    def test_index_out_of_range_raises(self):
        X = np.ones((10, 3))
        with pytest.raises(ValueError, match="out of range"):
            partial_correlation(X, 0, 5)

    def test_perfect_correlation(self):
        n = 100
        x = np.linspace(0, 10, n)
        X = np.column_stack([x, 2 * x + 1, np.random.default_rng(0).normal(0, 1, n)])
        r = partial_correlation(X, 0, 1)
        assert r == pytest.approx(1.0, abs=1e-10)


class TestPartialCorrelationMatrix:

    def test_diagonal_is_one(self, rng):
        X = rng.normal(0, 1, (100, 4))
        pcor = partial_correlation_matrix(X)
        np.testing.assert_array_almost_equal(np.diag(pcor), np.ones(4))

    def test_symmetric(self, rng):
        X = rng.normal(0, 1, (100, 4))
        pcor = partial_correlation_matrix(X)
        np.testing.assert_array_almost_equal(pcor, pcor.T, decimal=10)

    def test_shape(self, rng):
        X = rng.normal(0, 1, (50, 5))
        pcor = partial_correlation_matrix(X)
        assert pcor.shape == (5, 5)


# ===================================================================
# Fisher z-test
# ===================================================================


class TestFisherZTest:

    def test_zero_correlation_high_pvalue(self):
        z_stat, p_val = fisher_z_test(0.0, 100, k=0)
        assert z_stat == pytest.approx(0.0, abs=1e-8)
        assert p_val == pytest.approx(1.0, abs=1e-8)

    def test_strong_correlation_low_pvalue(self):
        _, p_val = fisher_z_test(0.9, 100, k=0)
        assert p_val < 0.001

    def test_insufficient_df_returns_default(self):
        z_stat, p_val = fisher_z_test(0.5, 5, k=3)
        assert z_stat == 0.0
        assert p_val == 1.0

    @pytest.mark.parametrize("n", [20, 50, 200])
    def test_pvalue_decreases_with_n(self, n):
        _, p1 = fisher_z_test(0.3, n, k=0)
        _, p2 = fisher_z_test(0.3, n * 2, k=0)
        assert p2 <= p1


# ===================================================================
# CI testing
# ===================================================================


class TestPartialCorrelationTest:

    def test_independent_variables_not_rejected(self, rng):
        X = rng.normal(0, 1, (200, 3))
        r, p_val, is_indep = partial_correlation_test(X, 0, 1, alpha=0.05)
        assert is_indep  # should not reject independence

    def test_dependent_variables_rejected(self, rng):
        n = 500
        X0 = rng.normal(0, 1, n)
        X1 = 3.0 * X0 + 0.1 * rng.normal(0, 1, n)
        X = np.column_stack([X0, X1, rng.normal(0, 1, n)])
        r, p_val, is_indep = partial_correlation_test(X, 0, 1, alpha=0.05)
        assert not is_indep
        assert p_val < 0.05

    def test_conditional_independence_detected(self, rng):
        n = 1000
        Z = rng.normal(0, 1, n)
        X0 = Z + 0.1 * rng.normal(0, 1, n)
        X1 = Z + 0.1 * rng.normal(0, 1, n)
        X = np.column_stack([X0, X1, Z])
        r, p_val, is_indep = partial_correlation_test(X, 0, 1, [2], alpha=0.05)
        assert is_indep  # independent given Z


# ===================================================================
# Multiple testing corrections
# ===================================================================


class TestBonferroni:

    def test_empty(self):
        adj, rej = bonferroni_correction(np.array([]))
        assert len(adj) == 0
        assert len(rej) == 0

    def test_single_significant(self):
        adj, rej = bonferroni_correction(np.array([0.01]), alpha=0.05)
        assert adj[0] == pytest.approx(0.01)
        assert rej[0]

    def test_multiplied_by_m(self):
        p = np.array([0.01, 0.02, 0.03])
        adj, _ = bonferroni_correction(p, alpha=0.05)
        np.testing.assert_array_almost_equal(adj, [0.03, 0.06, 0.09])

    def test_capped_at_one(self):
        p = np.array([0.5, 0.6])
        adj, _ = bonferroni_correction(p)
        assert np.all(adj <= 1.0)

    def test_more_conservative_than_bh(self):
        p = np.array([0.01, 0.03, 0.04, 0.08, 0.10])
        _, rej_bonf = bonferroni_correction(p, alpha=0.05)
        _, rej_bh = bh_fdr_correction(p, alpha=0.05)
        assert np.sum(rej_bonf) <= np.sum(rej_bh)


class TestBHFDR:

    def test_empty(self):
        adj, rej = bh_fdr_correction(np.array([]))
        assert len(adj) == 0

    def test_all_significant(self):
        p = np.array([0.001, 0.002, 0.003])
        _, rej = bh_fdr_correction(p, alpha=0.05)
        assert np.all(rej)

    def test_adjusted_monotone(self):
        """Adjusted p-values should be non-decreasing when sorted."""
        p = np.array([0.005, 0.01, 0.03, 0.04, 0.50])
        adj, _ = bh_fdr_correction(p, alpha=0.05)
        sorted_adj = adj[np.argsort(p)]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-12

    def test_adjusted_capped_at_one(self):
        p = np.array([0.3, 0.7, 0.9])
        adj, _ = bh_fdr_correction(p)
        assert np.all(adj <= 1.0)


# ===================================================================
# Bootstrap
# ===================================================================


class TestBootstrap:

    def test_ci_contains_true_mean(self, rng):
        data = rng.normal(5.0, 1.0, 200)
        point, lower, upper = bootstrap_ci(
            data, np.mean, n_bootstrap=2000, confidence=0.95, rng=rng,
        )
        assert lower < 5.0 < upper

    def test_point_estimate_is_statistic(self, rng):
        data = rng.normal(0, 1, 100)
        point, _, _ = bootstrap_ci(data, np.mean, n_bootstrap=500, rng=rng)
        assert point == pytest.approx(np.mean(data))

    def test_higher_confidence_wider_interval(self, rng):
        data = rng.normal(0, 1, 100)
        _, lo90, hi90 = bootstrap_ci(
            data, np.mean, n_bootstrap=2000, confidence=0.90, rng=rng,
        )
        _, lo99, hi99 = bootstrap_ci(
            data, np.mean, n_bootstrap=2000, confidence=0.99, rng=rng,
        )
        assert (hi99 - lo99) >= (hi90 - lo90) - 0.05  # allow small tolerance

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([]), np.mean)

    def test_unknown_method_raises(self, rng):
        data = rng.normal(0, 1, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_ci(data, np.mean, method="invalid", rng=rng)

    def test_median_bootstrap(self, rng):
        data = rng.normal(3.0, 1.0, 300)
        point, lower, upper = bootstrap_ci(
            data, np.median, n_bootstrap=1000, rng=rng,
        )
        assert lower < 3.0 < upper
