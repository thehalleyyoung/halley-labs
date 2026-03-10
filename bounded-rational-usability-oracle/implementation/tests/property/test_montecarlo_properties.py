"""Property-based tests for the Monte Carlo sampling module.

Verifies statistical invariants: WLLN convergence, unbiased variance
estimation, effective sample size bounds, antithetic variate correlation,
CDF monotonicity, and quantile ordering.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats,
    integers,
    lists,
    composite,
)

from usability_oracle.montecarlo.diagnostics import MCDiagnostics
from usability_oracle.montecarlo.variance_reduction import AntitheticVariates
from usability_oracle.montecarlo.statistics import TrajectoryStatistics

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_pos_float = floats(min_value=0.01, max_value=100.0,
                    allow_nan=False, allow_infinity=False)

_sample_size = integers(min_value=20, max_value=500)


@composite
def _weight_vector(draw, min_size=5, max_size=50):
    """Generate a valid importance weight vector (positive, sums to 1)."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    raw = draw(lists(
        floats(min_value=0.001, max_value=10.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    total = sum(raw)
    assume(total > 0)
    return np.array([x / total for x in raw])


@composite
def _sample_array(draw, min_size=20, max_size=200):
    """Generate an array of floating-point samples."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    vals = draw(lists(
        floats(min_value=-100.0, max_value=100.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    return np.array(vals)


_ATOL = 1e-6

# ---------------------------------------------------------------------------
# Mean of large sample converges to true mean (WLLN)
# ---------------------------------------------------------------------------


def test_wlln_convergence_normal():
    """Sample mean converges to population mean for large N (Gaussian)."""
    rng = np.random.default_rng(42)
    true_mean = 5.0
    n = 10000
    samples = rng.normal(loc=true_mean, scale=2.0, size=n)
    sample_mean = np.mean(samples)
    assert abs(sample_mean - true_mean) < 0.1, \
        f"Sample mean {sample_mean} far from true mean {true_mean}"


def test_wlln_convergence_uniform():
    """Sample mean converges to population mean for uniform distribution."""
    rng = np.random.default_rng(123)
    a, b = 2.0, 8.0
    true_mean = (a + b) / 2.0
    n = 10000
    samples = rng.uniform(a, b, size=n)
    sample_mean = np.mean(samples)
    assert abs(sample_mean - true_mean) < 0.1, \
        f"Sample mean {sample_mean} far from true mean {true_mean}"


@given(floats(min_value=-10.0, max_value=10.0,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.1, max_value=5.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_wlln_parametric(mu, sigma):
    """WLLN for arbitrary Gaussian(μ, σ)."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=mu, scale=sigma, size=5000)
    sample_mean = np.mean(samples)
    tolerance = 3 * sigma / math.sqrt(5000)
    assert abs(sample_mean - mu) < max(tolerance, 0.2), \
        f"|{sample_mean} - {mu}| > {tolerance}"


# ---------------------------------------------------------------------------
# Variance estimator is unbiased
# ---------------------------------------------------------------------------


def test_variance_estimator_unbiased():
    """Sample variance with ddof=1 is unbiased: E[S²] ≈ σ²."""
    rng = np.random.default_rng(42)
    true_var = 4.0
    n_trials = 2000
    n_samples = 50
    var_estimates = []
    for _ in range(n_trials):
        samples = rng.normal(0, math.sqrt(true_var), size=n_samples)
        var_estimates.append(np.var(samples, ddof=1))
    mean_var = np.mean(var_estimates)
    assert abs(mean_var - true_var) < 0.3, \
        f"Mean variance estimate {mean_var} far from {true_var}"


# ---------------------------------------------------------------------------
# Effective sample size ≤ n
# ---------------------------------------------------------------------------


@given(_weight_vector())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_ess_leq_n(weights):
    """ESS ≤ n for any weight vector."""
    n = len(weights)
    samples = np.random.default_rng(42).standard_normal(n)
    ess = MCDiagnostics.effective_sample_size(samples, weights=list(weights))
    assert ess <= n + _ATOL, f"ESS {ess} > n {n}"
    assert ess >= 0, f"ESS is negative: {ess}"


# ---------------------------------------------------------------------------
# ESS = n for uniform weights
# ---------------------------------------------------------------------------


@given(integers(min_value=5, max_value=100))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_ess_equals_n_for_uniform_weights(n):
    """When all weights are equal, ESS = n."""
    weights = np.ones(n) / n
    samples = np.random.default_rng(42).standard_normal(n)
    ess = MCDiagnostics.effective_sample_size(samples, weights=list(weights))
    assert math.isclose(ess, n, rel_tol=1e-4), \
        f"ESS for uniform weights should be {n}, got {ess}"


# ---------------------------------------------------------------------------
# ESS = 1 for degenerate weights
# ---------------------------------------------------------------------------


@given(integers(min_value=5, max_value=100))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_ess_equals_one_for_degenerate_weights(n):
    """When one weight dominates, ESS → 1."""
    weights = np.zeros(n)
    weights[0] = 1.0
    samples = np.random.default_rng(42).standard_normal(n)
    ess = MCDiagnostics.effective_sample_size(samples, weights=list(weights))
    assert math.isclose(ess, 1.0, abs_tol=1e-4), \
        f"ESS for degenerate weights should be 1, got {ess}"


# ---------------------------------------------------------------------------
# Antithetic variates have negative correlation
# ---------------------------------------------------------------------------


def test_antithetic_variates_negative_correlation():
    """Antithetic variates should produce negatively correlated estimates."""
    rng = np.random.default_rng(42)
    n = 1000
    u = rng.uniform(0, 1, size=n)
    # f(x) = x^2 — a simple increasing function
    originals = u ** 2
    antithetics = (1 - u) ** 2
    correlation = np.corrcoef(originals, antithetics)[0, 1]
    assert correlation < 0, \
        f"Antithetic correlation should be negative, got {correlation}"


def test_antithetic_variates_pair_estimates():
    """AntitheticVariates.pair_estimates reduces variance."""
    originals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    antithetics = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    paired, ratio = AntitheticVariates().pair_estimates(originals, antithetics)
    assert len(paired) == len(originals)
    # Paired average should be constant (3.0 for all)
    for v in paired:
        assert math.isclose(v, 3.0, abs_tol=_ATOL)


# ---------------------------------------------------------------------------
# CDF is monotonically non-decreasing
# ---------------------------------------------------------------------------


def test_cdf_monotone_nondecreasing():
    """Empirical CDF must be monotonically non-decreasing."""
    rng = np.random.default_rng(42)
    costs = rng.exponential(2.0, size=100)
    sorted_costs = np.sort(costs)
    n = len(sorted_costs)
    cdf_values = np.arange(1, n + 1) / n
    for i in range(1, len(cdf_values)):
        assert cdf_values[i] >= cdf_values[i - 1], \
            f"CDF not monotone at index {i}: {cdf_values[i-1]} > {cdf_values[i]}"


@given(_sample_array(min_size=10, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_cdf_values_in_unit_interval(samples):
    """CDF values are in [0, 1]."""
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    cdf = np.arange(1, n + 1) / n
    assert np.all(cdf >= 0.0)
    assert np.all(cdf <= 1.0 + _ATOL)


# ---------------------------------------------------------------------------
# Quantiles are ordered: q(0.25) ≤ q(0.5) ≤ q(0.75)
# ---------------------------------------------------------------------------


@given(_sample_array(min_size=20, max_size=200))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_quantiles_ordered(samples):
    """Quantiles must be ordered: q(0.25) ≤ q(0.5) ≤ q(0.75)."""
    q25 = np.percentile(samples, 25)
    q50 = np.percentile(samples, 50)
    q75 = np.percentile(samples, 75)
    assert q25 <= q50 + _ATOL, f"q25={q25} > q50={q50}"
    assert q50 <= q75 + _ATOL, f"q50={q50} > q75={q75}"


@given(_sample_array(min_size=20, max_size=100))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_quantile_sequence_non_decreasing(samples):
    """Full quantile sequence must be non-decreasing."""
    quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantiles = [np.percentile(samples, q * 100) for q in quantile_levels]
    for i in range(1, len(quantiles)):
        assert quantiles[i] >= quantiles[i - 1] - _ATOL, \
            f"Quantile at {quantile_levels[i]} < {quantile_levels[i-1]}"


# ---------------------------------------------------------------------------
# Gelman-Rubin diagnostic
# ---------------------------------------------------------------------------


def test_gelman_rubin_converged_chains():
    """R-hat ≈ 1 for converged chains from the same distribution."""
    rng = np.random.default_rng(42)
    chains = [list(rng.normal(0, 1, size=500)) for _ in range(4)]
    r_hat = MCDiagnostics.gelman_rubin_diagnostic(chains)
    assert r_hat < 1.1, f"R-hat for converged chains should be < 1.1, got {r_hat}"


def test_gelman_rubin_diverged_chains():
    """R-hat >> 1 for chains from different distributions."""
    rng = np.random.default_rng(42)
    chains = [
        list(rng.normal(0, 1, size=200)),
        list(rng.normal(10, 1, size=200)),
    ]
    r_hat = MCDiagnostics.gelman_rubin_diagnostic(chains)
    assert r_hat > 1.5, f"R-hat for diverged chains should be >> 1, got {r_hat}"


# ---------------------------------------------------------------------------
# Autocorrelation at lag 0 is 1
# ---------------------------------------------------------------------------


def test_autocorrelation_lag_zero_is_one():
    """Autocorrelation at lag 0 equals 1."""
    rng = np.random.default_rng(42)
    chain = list(rng.normal(0, 1, size=100))
    acf = MCDiagnostics.autocorrelation(chain, max_lag=5)
    assert math.isclose(acf[0], 1.0, abs_tol=1e-4), \
        f"Autocorrelation at lag 0 should be 1, got {acf[0]}"
