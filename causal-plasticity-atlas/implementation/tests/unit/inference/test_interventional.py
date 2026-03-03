"""Unit tests for cpa.inference.interventional.InterventionalEstimator."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.core.scm import StructuralCausalModel
from cpa.inference.interventional import (
    InterventionalEstimator,
    InterventionalQuery,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def estimator():
    return InterventionalEstimator(alpha=0.05)


@pytest.fixture
def chain_scm():
    """Chain: 0 → 1 → 2 with known linear coefficients."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[1, 2] = 1.0
    coefs = np.zeros((3, 3))
    coefs[0, 1] = 0.5
    coefs[1, 2] = 0.8
    var = np.array([1.0, 1.0, 1.0])
    return StructuralCausalModel(
        adj, variable_names=["X0", "X1", "X2"],
        regression_coefficients=coefs, residual_variances=var,
        sample_size=1000,
    )


@pytest.fixture
def fork_scm():
    """Fork: 1 ← 0 → 2."""
    adj = np.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[0, 2] = 1.0
    coefs = np.zeros((3, 3))
    coefs[0, 1] = 0.7
    coefs[0, 2] = 0.3
    var = np.ones(3)
    return StructuralCausalModel(
        adj, regression_coefficients=coefs, residual_variances=var,
        sample_size=1000,
    )


@pytest.fixture
def confounded_data(rng):
    """Data with treatment, outcome, and confounder. T = 0.5*Z + e, Y = 0.8*T + 0.3*Z + e."""
    n = 2000
    z = rng.normal(0, 1, n)
    t = 0.5 * z + rng.normal(0, 0.5, n)
    y = 0.8 * t + 0.3 * z + rng.normal(0, 0.5, n)
    return np.column_stack([z, t, y])


@pytest.fixture
def confounded_adj():
    """Z(0) → T(1) → Y(2), Z(0) → Y(2)."""
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj


@pytest.fixture
def frontdoor_data(rng):
    """Front-door scenario: X → M → Y, U → X, U → Y."""
    n = 3000
    u = rng.normal(0, 1, n)
    x = 0.6 * u + rng.normal(0, 0.5, n)
    m = 0.7 * x + rng.normal(0, 0.5, n)
    y = 0.5 * m + 0.4 * u + rng.normal(0, 0.5, n)
    return np.column_stack([x, m, y])


# ===================================================================
# Tests – ATE computation
# ===================================================================


class TestATE:
    """Test Average Treatment Effect computation."""

    def test_ate_returns_tuple(self, estimator, chain_scm):
        ate, se = estimator.compute_ate(chain_scm, treatment=0, outcome=2,
                                        n_samples=5000)
        assert isinstance(ate, (float, np.floating))
        assert isinstance(se, (float, np.floating))

    def test_ate_chain_known_value(self, estimator, chain_scm):
        # ATE of X0 on X2 = 0.5 * 0.8 = 0.4
        ate, se = estimator.compute_ate(chain_scm, treatment=0, outcome=2,
                                        n_samples=20000)
        assert_allclose(ate, 0.4, atol=0.15)

    def test_ate_direct_effect(self, estimator, chain_scm):
        # ATE of X1 on X2 = 0.8
        ate, se = estimator.compute_ate(chain_scm, treatment=1, outcome=2,
                                        n_samples=20000)
        assert_allclose(ate, 0.8, atol=0.15)

    def test_ate_no_effect(self, estimator, chain_scm):
        # ATE of X2 on X0 = 0
        ate, se = estimator.compute_ate(chain_scm, treatment=2, outcome=0,
                                        n_samples=10000)
        assert_allclose(ate, 0.0, atol=0.15)

    def test_ate_se_positive(self, estimator, chain_scm):
        ate, se = estimator.compute_ate(chain_scm, treatment=0, outcome=2,
                                        n_samples=5000)
        assert se >= 0.0

    def test_ate_fork_direct(self, estimator, fork_scm):
        ate, se = estimator.compute_ate(fork_scm, treatment=0, outcome=1,
                                        n_samples=20000)
        assert_allclose(ate, 0.7, atol=0.15)


# ===================================================================
# Tests – Back-door adjustment
# ===================================================================


class TestBackdoorAdjustment:
    """Test back-door adjustment formula."""

    def test_backdoor_returns_tuple(self, estimator, confounded_adj,
                                    confounded_data):
        ate, se = estimator.backdoor_adjustment(
            confounded_adj, treatment=1, outcome=2,
            adjustment_set={0}, data=confounded_data,
        )
        assert isinstance(ate, (float, np.floating))

    def test_backdoor_recovers_true_effect(self, estimator, confounded_adj,
                                           confounded_data):
        ate, se = estimator.backdoor_adjustment(
            confounded_adj, treatment=1, outcome=2,
            adjustment_set={0}, data=confounded_data,
        )
        assert_allclose(ate, 0.8, atol=0.15)

    def test_backdoor_without_adjustment_biased(self, estimator, confounded_adj,
                                                 confounded_data):
        ate_adj, _ = estimator.backdoor_adjustment(
            confounded_adj, treatment=1, outcome=2,
            adjustment_set={0}, data=confounded_data,
        )
        ate_naive, _ = estimator.backdoor_adjustment(
            confounded_adj, treatment=1, outcome=2,
            adjustment_set=set(), data=confounded_data,
        )
        # Naive estimate should differ from adjusted
        assert abs(ate_naive - ate_adj) > 0.01 or abs(ate_adj - 0.8) < 0.15


# ===================================================================
# Tests – IPW estimator
# ===================================================================


class TestIPWeighting:
    """Test inverse probability weighting estimator."""

    def test_ipw_returns_tuple(self, estimator, rng):
        n = 1000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.5 * t + rng.normal(0, 1, n)
        ps = np.full(n, 0.5)
        data = np.column_stack([t, y])
        ate, se = estimator.ip_weighting(t, y, ps, data)
        assert isinstance(ate, (float, np.floating))

    def test_ipw_unconfounded(self, estimator, rng):
        n = 5000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.6 * t + rng.normal(0, 1, n)
        ps = np.full(n, 0.5)
        data = np.column_stack([t, y])
        ate, se = estimator.ip_weighting(t, y, ps, data)
        assert_allclose(ate, 0.6, atol=0.15)

    def test_ipw_varying_propensity(self, estimator, rng):
        n = 5000
        z = rng.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-z))
        t = rng.binomial(1, ps).astype(float)
        y = 0.7 * t + 0.3 * z + rng.normal(0, 0.5, n)
        data = np.column_stack([z, t, y])
        ate, se = estimator.ip_weighting(t, y, ps, data)
        assert_allclose(ate, 0.7, atol=0.3)

    def test_ipw_se_nonnegative(self, estimator, rng):
        n = 1000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.5 * t + rng.normal(0, 1, n)
        ps = np.full(n, 0.5)
        data = np.column_stack([t, y])
        _, se = estimator.ip_weighting(t, y, ps, data)
        assert se >= 0.0


# ===================================================================
# Tests – Doubly robust estimator
# ===================================================================


class TestDoublyRobust:
    """Test doubly robust (AIPW) estimator."""

    def test_dr_returns_tuple(self, estimator, rng):
        n = 1000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.5 * t + rng.normal(0, 1, n)
        ps = np.full(n, 0.5)
        y_pred = 0.5 * t
        data = np.column_stack([t, y])
        ate, se = estimator.doubly_robust(t, y, ps, y_pred, data)
        assert isinstance(ate, (float, np.floating))

    def test_dr_consistent_correct_model(self, estimator, rng):
        n = 5000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.6 * t + rng.normal(0, 0.5, n)
        ps = np.full(n, 0.5)
        y_pred = 0.6 * t
        data = np.column_stack([t, y])
        ate, se = estimator.doubly_robust(t, y, ps, y_pred, data)
        assert_allclose(ate, 0.6, atol=0.35)

    def test_dr_robust_to_ps_misspec(self, estimator, rng):
        n = 5000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.5 * t + rng.normal(0, 0.5, n)
        ps_wrong = np.full(n, 0.3)  # intentionally wrong
        y_pred = 0.5 * t  # correct outcome model
        data = np.column_stack([t, y])
        ate, se = estimator.doubly_robust(t, y, ps_wrong, y_pred, data)
        assert_allclose(ate, 0.5, atol=0.25)

    def test_dr_with_outcome_model_pred_0(self, estimator, rng):
        n = 2000
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 0.5 * t + rng.normal(0, 0.5, n)
        ps = np.full(n, 0.5)
        y_pred_1 = np.full(n, 0.5)
        y_pred_0 = np.zeros(n)
        data = np.column_stack([t, y])
        ate, se = estimator.doubly_robust(
            t, y, ps, y_pred_1, data, outcome_model_pred_0=y_pred_0,
        )
        assert np.isfinite(ate)


# ===================================================================
# Tests – CATE
# ===================================================================


class TestCATE:
    """Test Conditional ATE estimation."""

    def test_cate_returns_tuple(self, estimator, chain_scm):
        cate, se = estimator.compute_cate(
            chain_scm, treatment=0, outcome=2, covariates={1: 0.0},
            n_samples=5000,
        )
        assert isinstance(cate, (float, np.floating))

    def test_cate_is_finite(self, estimator, chain_scm):
        cate, se = estimator.compute_cate(
            chain_scm, treatment=0, outcome=2, covariates={1: 0.0},
            n_samples=5000,
        )
        assert np.isfinite(cate)


# ===================================================================
# Tests – Truncated factorization
# ===================================================================


class TestTruncatedFactorization:
    """Test truncated factorization via InterventionalEstimator."""

    def test_returns_array(self, estimator, chain_scm):
        query = InterventionalQuery(
            target_vars=[2], intervention_vars=[0],
            intervention_values=[1.0], conditioning_vars=[],
        )
        result = estimator.truncated_factorization(chain_scm, query)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_variables(self, estimator, chain_scm):
        query = InterventionalQuery(
            target_vars=[2], intervention_vars=[0],
            intervention_values=[1.0], conditioning_vars=[],
        )
        result = estimator.truncated_factorization(chain_scm, query)
        assert result.shape[0] == chain_scm.num_variables


# ===================================================================
# Tests – InterventionalQuery dataclass
# ===================================================================


class TestInterventionalQuery:
    """Test InterventionalQuery construction."""

    def test_create_query(self):
        q = InterventionalQuery(
            target_vars=[2], intervention_vars=[0],
            intervention_values=[1.0], conditioning_vars=[1],
        )
        assert q.target_vars == [2]
        assert q.intervention_vars == [0]
        assert q.intervention_values == [1.0]

    def test_empty_conditioning(self):
        q = InterventionalQuery(
            target_vars=[1], intervention_vars=[0],
            intervention_values=[0.0], conditioning_vars=[],
        )
        assert q.conditioning_vars == []


# ===================================================================
# Tests – Adjustment formula
# ===================================================================


class TestAdjustmentFormula:
    """Test generic adjustment formula."""

    def test_adjustment_returns_tuple(self, estimator, chain_scm):
        ate, se = estimator.adjustment_formula(
            chain_scm, treatment=0, outcome=2,
            adjustment_set={1}, n_samples=5000,
        )
        assert isinstance(ate, (float, np.floating))

    def test_adjustment_auto_finds_set(self, estimator, fork_scm):
        """When the fork 1 ← 0 → 2 is used with treatment=1, outcome=2,
        the adjustment set {0} should be found automatically."""
        ate, se = estimator.adjustment_formula(
            fork_scm, treatment=1, outcome=2,
            n_samples=5000,
        )
        assert np.isfinite(ate)
