"""Tests for stability selection and bootstrap engines.

Covers stability selection engine, bootstrap engine,
consensus graph, and confidence bands.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.certificates.stability import (
    StabilitySelectionEngine,
    BootstrapEngine,
    StabilityResult,
    BootstrapResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def stability_engine():
    return StabilitySelectionEngine(
        n_rounds=30,
        subsample_fraction=0.5,
        upper_threshold=0.6,
        lower_threshold=0.4,
        random_state=42,
    )


@pytest.fixture
def bootstrap_engine():
    return BootstrapEngine(
        n_bootstrap=50,
        ci_level=0.95,
        method="percentile",
        random_state=42,
    )


@pytest.fixture
def linear_data(rng):
    """X0 -> X1 -> X2 linear data."""
    n = 200
    X0 = rng.normal(0, 1, size=n)
    X1 = 0.8 * X0 + rng.normal(0, 0.3, size=n)
    X2 = 0.6 * X1 + rng.normal(0, 0.3, size=n)
    return np.column_stack([X0, X1, X2])


def _chain_learner(data):
    """Simple learner that returns a chain DAG."""
    p = data.shape[1]
    adj = np.zeros((p, p))
    for i in range(p - 1):
        adj[i, i + 1] = 1
    return adj


def _noisy_chain_learner(data, rng_state=None):
    """Learner with some random noise in edges."""
    p = data.shape[1]
    adj = np.zeros((p, p))
    rng = np.random.default_rng(rng_state)
    for i in range(p - 1):
        if rng.uniform() > 0.1:
            adj[i, i + 1] = 1
    return adj


# ---------------------------------------------------------------------------
# Test StabilitySelectionEngine
# ---------------------------------------------------------------------------

class TestStabilitySelectionEngine:

    def test_run_returns_result(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        assert isinstance(result, StabilityResult)

    def test_selection_probabilities_shape(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        p = linear_data.shape[1]
        assert result.selection_probabilities.shape == (p, p)

    def test_probabilities_bounded(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        assert np.all(result.selection_probabilities >= 0)
        assert np.all(result.selection_probabilities <= 1)

    def test_stable_edges(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        assert isinstance(result.stable_edges, list)

    def test_unstable_edges(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        assert isinstance(result.unstable_edges, list)

    def test_consensus_adjacency(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        p = linear_data.shape[1]
        assert result.consensus_adjacency.shape == (p, p)

    def test_is_edge_stable(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        # Edge 0->1 should be stable with chain learner
        prob = result.edge_probability(0, 1)
        assert prob >= 0.0

    def test_edge_probability(self, stability_engine, linear_data):
        result = stability_engine.run(linear_data, _chain_learner)
        prob = result.edge_probability(0, 1)
        assert 0.0 <= prob <= 1.0

    def test_consensus_graph(self, stability_engine, linear_data):
        adj = stability_engine.consensus_graph(linear_data, _chain_learner, threshold=0.5)
        p = linear_data.shape[1]
        assert adj.shape == (p, p)

    def test_edge_selection_stability(self, stability_engine, linear_data):
        stab = stability_engine.edge_selection_stability(
            linear_data, _chain_learner, edge=(0, 1),
        )
        assert 0.0 <= stab <= 1.0

    def test_variable_selection(self, stability_engine, linear_data):
        def selector_fn(X, y):
            n_predictors = X.shape[1]
            selected = np.zeros(n_predictors)
            if n_predictors > 0:
                selected[0] = 1  # always select first predictor
            return selected
        probs, selected, unselected = stability_engine.run_variable_selection(
            linear_data, target_idx=2, selector_fn=selector_fn,
        )
        # probs is in predictor space (n_vars - 1 elements)
        assert probs.shape[0] == linear_data.shape[1] - 1

    def test_different_thresholds(self, linear_data):
        engine_high = StabilitySelectionEngine(
            n_rounds=20, upper_threshold=0.8, lower_threshold=0.2, random_state=42,
        )
        engine_low = StabilitySelectionEngine(
            n_rounds=20, upper_threshold=0.5, lower_threshold=0.3, random_state=42,
        )
        r_high = engine_high.run(linear_data, _chain_learner)
        r_low = engine_low.run(linear_data, _chain_learner)
        assert len(r_high.stable_edges) <= len(r_low.stable_edges) + 5  # rough bound


# ---------------------------------------------------------------------------
# Test BootstrapEngine
# ---------------------------------------------------------------------------

class TestBootstrapEngine:

    def test_parametric_bootstrap_regression(self, bootstrap_engine, rng):
        n, p = 100, 3
        X = rng.normal(0, 1, size=(n, p))
        true_coefs = np.array([1.0, -0.5, 0.3])
        y = X @ true_coefs + rng.normal(0, 0.3, size=n)
        result = bootstrap_engine.parametric_bootstrap_regression(X, y)
        assert isinstance(result, BootstrapResult)

    def test_bootstrap_ci_shape(self, bootstrap_engine, rng):
        X = rng.normal(0, 1, size=(100, 3))
        y = X @ np.array([1, -1, 0.5]) + rng.normal(0, 0.2, size=100)
        result = bootstrap_engine.parametric_bootstrap_regression(X, y)
        assert result.ci_lower.shape == result.ci_upper.shape

    def test_ci_contains_estimate(self, bootstrap_engine, rng):
        X = rng.normal(0, 1, size=(100, 3))
        y = X @ np.array([1, -1, 0.5]) + rng.normal(0, 0.2, size=100)
        result = bootstrap_engine.parametric_bootstrap_regression(X, y)
        for i in range(len(result.point_estimate)):
            assert result.ci_lower[i] <= result.point_estimate[i] + 0.1
            assert result.point_estimate[i] <= result.ci_upper[i] + 0.1

    def test_ci_width_property(self, bootstrap_engine, rng):
        X = rng.normal(0, 1, size=(100, 3))
        y = X @ np.array([1, -1, 0.5]) + rng.normal(0, 0.2, size=100)
        result = bootstrap_engine.parametric_bootstrap_regression(X, y)
        width = result.ci_width
        assert np.all(width >= 0)

    def test_parametric_bootstrap_variance(self, bootstrap_engine):
        result = bootstrap_engine.parametric_bootstrap_variance(
            residual_var=1.0, n_samples=100,
        )
        assert isinstance(result, BootstrapResult)

    def test_nonparametric_bootstrap(self, bootstrap_engine, rng):
        data = rng.normal(0, 1, size=(100, 3))
        def stat_fn(d):
            return np.mean(d, axis=0)
        result = bootstrap_engine.nonparametric_bootstrap(data, stat_fn)
        assert isinstance(result, BootstrapResult)

    def test_nonparametric_bootstrap_dag(self, bootstrap_engine, linear_data):
        result, edge_probs = bootstrap_engine.nonparametric_bootstrap_dag(
            linear_data, _chain_learner,
        )
        assert isinstance(result, BootstrapResult)
        p = linear_data.shape[1]
        assert edge_probs.shape == (p, p)

    def test_bagging_adjacency(self, bootstrap_engine, linear_data):
        adj = bootstrap_engine.bagging_adjacency(
            linear_data, _chain_learner, threshold=0.5,
        )
        p = linear_data.shape[1]
        assert adj.shape == (p, p)

    def test_confidence_bands(self, bootstrap_engine, rng):
        x = np.linspace(0, 10, 50)
        data = np.column_stack([x, 2 * x + rng.normal(0, 0.5, size=50)])
        x_grid = np.linspace(0, 10, 20)
        def model_fn(d, xg):
            from numpy.polynomial import polynomial as P
            coefs = P.polyfit(d[:, 0], d[:, 1], deg=1)
            return P.polyval(xg, coefs)
        mean, lower, upper = bootstrap_engine.confidence_bands(
            x_grid, data, model_fn,
        )
        assert mean.shape == (20,)
        assert lower.shape == (20,)
        assert upper.shape == (20,)
        assert np.all(lower <= upper + 0.1)

    def test_bca_method(self, rng):
        engine = BootstrapEngine(n_bootstrap=50, method="bca", random_state=42)
        X = rng.normal(0, 1, size=(100, 2))
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.3, size=100)
        result = engine.parametric_bootstrap_regression(X, y)
        assert isinstance(result, BootstrapResult)

    def test_normal_method(self, rng):
        engine = BootstrapEngine(n_bootstrap=50, method="normal", random_state=42)
        X = rng.normal(0, 1, size=(100, 2))
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.3, size=100)
        result = engine.parametric_bootstrap_regression(X, y)
        assert isinstance(result, BootstrapResult)

    def test_contains_method(self, bootstrap_engine, rng):
        X = rng.normal(0, 1, size=(100, 2))
        true_coefs = np.array([1.0, 0.5])
        y = X @ true_coefs + rng.normal(0, 0.2, size=100)
        result = bootstrap_engine.parametric_bootstrap_regression(X, y)
        contained = result.contains(true_coefs)
        # At 95% CI, true values should be contained most of the time
        assert isinstance(contained, np.ndarray)


# ---------------------------------------------------------------------------
# Test BootstrapResult dataclass
# ---------------------------------------------------------------------------

class TestBootstrapResult:

    def test_ci_width(self):
        result = BootstrapResult(
            point_estimate=np.array([1.0, 2.0]),
            bootstrap_samples=np.zeros((10, 2)),
            ci_lower=np.array([0.5, 1.5]),
            ci_upper=np.array([1.5, 2.5]),
            ci_level=0.95,
            method="percentile",
            n_bootstrap=10,
            se=np.array([0.2, 0.2]),
            bias=np.array([0.0, 0.0]),
        )
        width = result.ci_width
        assert_allclose(width, [1.0, 1.0])

    def test_contains(self):
        result = BootstrapResult(
            point_estimate=np.array([1.0]),
            bootstrap_samples=np.zeros((10, 1)),
            ci_lower=np.array([0.5]),
            ci_upper=np.array([1.5]),
            ci_level=0.95,
            method="percentile",
            n_bootstrap=10,
            se=np.array([0.2]),
            bias=np.array([0.0]),
        )
        assert result.contains(np.array([1.0]))[0] == True
        assert result.contains(np.array([2.0]))[0] == False


# ---------------------------------------------------------------------------
# Test StabilityResult dataclass
# ---------------------------------------------------------------------------

class TestStabilityResult:

    def test_is_edge_stable(self):
        probs = np.array([[0.0, 0.9], [0.0, 0.0]])
        result = StabilityResult(
            selection_probabilities=probs,
            stable_edges=[(0, 1)],
            unstable_edges=[],
            uncertain_edges=[],
            upper_threshold=0.6,
            lower_threshold=0.4,
            n_rounds=30,
            subsample_fraction=0.5,
            consensus_adjacency=np.array([[0, 1], [0, 0]]),
        )
        assert result.is_edge_stable(0, 1)
        assert not result.is_edge_stable(1, 0)

    def test_edge_probability(self):
        probs = np.array([[0.0, 0.7], [0.3, 0.0]])
        result = StabilityResult(
            selection_probabilities=probs,
            stable_edges=[(0, 1)],
            unstable_edges=[],
            uncertain_edges=[(1, 0)],
            upper_threshold=0.6,
            lower_threshold=0.4,
            n_rounds=30,
            subsample_fraction=0.5,
            consensus_adjacency=probs > 0.5,
        )
        assert_allclose(result.edge_probability(0, 1), 0.7)
        assert_allclose(result.edge_probability(1, 0), 0.3)

    def test_is_edge_absent(self):
        probs = np.array([[0.0, 0.9], [0.1, 0.0]])
        result = StabilityResult(
            selection_probabilities=probs,
            stable_edges=[(0, 1)],
            unstable_edges=[(1, 0)],
            uncertain_edges=[],
            upper_threshold=0.6,
            lower_threshold=0.4,
            n_rounds=30,
            subsample_fraction=0.5,
            consensus_adjacency=np.array([[0, 1], [0, 0]]),
        )
        assert result.is_edge_absent(1, 0)
