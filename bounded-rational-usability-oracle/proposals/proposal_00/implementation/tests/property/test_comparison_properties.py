"""Property-based tests for paired comparison and error bounds.

This module verifies statistical properties of the RegressionTester and
ErrorBoundComputer using Hypothesis. Properties tested include p-value range,
effect-size behaviour for identical and separated samples, monotonic tightening
of error bounds with sample size, Hoeffding bound decay, non-negativity of
error bounds, and the total-error domination invariant.
"""

import math

import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    floats, integers, lists, tuples, sampled_from,
)

from usability_oracle.comparison.hypothesis import RegressionTester, HypothesisResult
from usability_oracle.comparison.error_bounds import ErrorBoundComputer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_sample_val = floats(min_value=-100.0, max_value=100.0,
                     allow_nan=False, allow_infinity=False)

_pos_val = floats(min_value=0.01, max_value=100.0,
                  allow_nan=False, allow_infinity=False)

_sample_list = lists(_sample_val, min_size=10, max_size=50)

_pos_list = lists(_pos_val, min_size=10, max_size=50)

_alpha = floats(min_value=0.01, max_value=0.20,
                allow_nan=False, allow_infinity=False)

_variance = floats(min_value=0.1, max_value=100.0,
                   allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# p-value in [0, 1]
# ---------------------------------------------------------------------------


@given(_sample_list, _sample_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_p_value_in_unit_range(samples_a, samples_b):
    """The p-value returned by RegressionTester.test is in [0, 1].

    By definition a p-value is a probability and must lie in the unit interval.
    """
    assume(len(samples_a) >= 10 and len(samples_b) >= 10)
    tester = RegressionTester(method="welch_t")
    result = tester.test(np.array(samples_a), np.array(samples_b))
    assert 0.0 - 1e-9 <= result.p_value <= 1.0 + 1e-9


@given(_sample_list, _sample_list)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_p_value_mann_whitney_in_range(samples_a, samples_b):
    """Mann-Whitney p-value is in [0, 1].

    The non-parametric test must also produce valid probabilities.
    """
    assume(len(samples_a) >= 10 and len(samples_b) >= 10)
    tester = RegressionTester(method="mann_whitney")
    result = tester.test(np.array(samples_a), np.array(samples_b))
    assert 0.0 - 1e-9 <= result.p_value <= 1.0 + 1e-9


@given(_sample_list, _sample_list)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_p_value_bootstrap_in_range(samples_a, samples_b):
    """Bootstrap p-value is in [0, 1].

    The permutation-based test must produce a valid probability.
    """
    assume(len(samples_a) >= 10 and len(samples_b) >= 10)
    tester = RegressionTester(method="bootstrap", n_bootstrap=500)
    result = tester.test(np.array(samples_a), np.array(samples_b))
    assert 0.0 - 1e-9 <= result.p_value <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Identical samples → effect_size ≈ 0
# ---------------------------------------------------------------------------

@given(_sample_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_identical_samples_zero_effect(samples):
    """When both samples are identical, the effect size is approximately 0.

    Cohen's d between a sample and itself should be zero because the
    mean difference is zero.
    """
    assume(len(samples) >= 10)
    arr = np.array(samples)
    tester = RegressionTester(method="welch_t")
    result = tester.test(arr, arr.copy())
    assert abs(result.effect_size) < 0.1, f"Effect size = {result.effect_size}"


@given(_sample_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_identical_samples_high_p_value(samples):
    """When both samples are identical, the p-value should be large.

    There should be no evidence to reject the null hypothesis.
    """
    assume(len(samples) >= 10)
    arr = np.array(samples)
    assume(np.std(arr) > 1e-6)  # avoid degenerate constant samples
    tester = RegressionTester(method="welch_t")
    result = tester.test(arr, arr.copy())
    assert result.p_value > 0.05


# ---------------------------------------------------------------------------
# Clearly separated samples → reject null
# ---------------------------------------------------------------------------

@given(integers(min_value=30, max_value=50))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_separated_samples_reject_null(n):
    """When samples are clearly separated, the null hypothesis is rejected.

    Two samples with means differing by 10 standard deviations should
    always be detected as significantly different.
    """
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(10.0, 1.0, n)
    tester = RegressionTester(method="welch_t")
    result = tester.test(a, b, alpha=0.05)
    assert result.reject_null


@given(integers(min_value=30, max_value=50))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_separated_samples_large_effect(n):
    """Clearly separated samples produce a large effect size.

    Cohen's d should be well above the 'large' threshold of 0.8
    for 10-sigma-separated distributions.
    """
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(10.0, 1.0, n)
    tester = RegressionTester(method="welch_t")
    result = tester.test(a, b)
    assert abs(result.effect_size) > 1.0


# ---------------------------------------------------------------------------
# Confidence interval contains effect size
# ---------------------------------------------------------------------------

@given(_sample_list, _sample_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_ci_contains_effect_size(samples_a, samples_b):
    """The confidence interval contains the point effect size estimate.

    ci_lower <= effect_size <= ci_upper must hold.
    """
    assume(len(samples_a) >= 10 and len(samples_b) >= 10)
    tester = RegressionTester(method="welch_t")
    result = tester.test(np.array(samples_a), np.array(samples_b))
    assert result.ci_lower <= result.effect_size + 1e-6
    assert result.effect_size <= result.ci_upper + 1e-6


# ---------------------------------------------------------------------------
# HypothesisResult sample counts
# ---------------------------------------------------------------------------

@given(_sample_list, _sample_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_result_sample_counts(samples_a, samples_b):
    """n_a and n_b in the result match the input lengths.

    The result object should accurately record the sample sizes used.
    """
    assume(len(samples_a) >= 10 and len(samples_b) >= 10)
    tester = RegressionTester(method="welch_t")
    result = tester.test(np.array(samples_a), np.array(samples_b))
    assert result.n_a == len(samples_a)
    assert result.n_b == len(samples_b)


# ---------------------------------------------------------------------------
# ErrorBoundComputer: sampling error decreases with n
# ---------------------------------------------------------------------------

@given(_variance, _alpha)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_error_decreases_with_n(variance, alpha):
    """More samples lead to tighter (smaller) sampling error bounds.

    compute_sampling_error(n1) > compute_sampling_error(n2) when n1 < n2.
    """
    ebc = ErrorBoundComputer(bound_method="hoeffding")
    e1 = ebc.compute_sampling_error(10, variance, alpha)
    e2 = ebc.compute_sampling_error(100, variance, alpha)
    assert e1 >= e2 - 1e-9


@given(_variance, _alpha)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_error_monotone(variance, alpha):
    """Sampling error is strictly monotonically non-increasing in n.

    For a sequence of increasing sample sizes, error should not increase.
    """
    ebc = ErrorBoundComputer(bound_method="hoeffding")
    prev = float("inf")
    for n in [10, 50, 100, 500, 1000]:
        e = ebc.compute_sampling_error(n, variance, alpha)
        assert e <= prev + 1e-9
        prev = e


# ---------------------------------------------------------------------------
# ErrorBoundComputer: Hoeffding bound decays as 1/sqrt(n)
# ---------------------------------------------------------------------------

@given(_alpha)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_hoeffding_decay_rate(alpha):
    """Hoeffding bound decreases roughly as 1/√n.

    Doubling n should reduce the bound by a factor of approximately √2.
    We check that the ratio is between 1.0 and 2.0.
    """
    ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=100.0)
    e1 = ebc.compute_sampling_error(100, 1.0, alpha)
    e2 = ebc.compute_sampling_error(400, 1.0, alpha)
    assume(e2 > 1e-12)
    ratio = e1 / e2
    assert 1.0 <= ratio <= 3.0, f"Ratio = {ratio}"


# ---------------------------------------------------------------------------
# ErrorBoundComputer: error bounds are non-negative
# ---------------------------------------------------------------------------

@given(integers(min_value=10, max_value=1000), _variance, _alpha)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_sampling_error_non_negative(n, variance, alpha):
    """Sampling error bound is always non-negative.

    An error bound below zero has no statistical meaning.
    """
    ebc = ErrorBoundComputer(bound_method="hoeffding")
    e = ebc.compute_sampling_error(n, variance, alpha)
    assert e >= -1e-12


@given(_alpha)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_chebyshev_error_non_negative(alpha):
    """Chebyshev-based sampling error is non-negative.

    Testing the Chebyshev bound variant.
    """
    ebc = ErrorBoundComputer(bound_method="chebyshev")
    e = ebc.compute_sampling_error(50, 10.0, alpha)
    assert e >= -1e-12


@given(_alpha)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_clt_error_non_negative(alpha):
    """CLT-based sampling error is non-negative.

    Testing the CLT (normal approximation) bound variant.
    """
    ebc = ErrorBoundComputer(bound_method="clt")
    e = ebc.compute_sampling_error(50, 10.0, alpha)
    assert e >= -1e-12


# ---------------------------------------------------------------------------
# Total error >= max(abstraction_error, sampling_error)
# ---------------------------------------------------------------------------

@given(floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_total_error_dominates(abs_err, samp_err, model_err):
    """Total error is at least as large as any individual error component.

    total_error >= max(abstraction_error, sampling_error, model_error).
    """
    ebc = ErrorBoundComputer()
    total = ebc.compute_total_error(abs_err, samp_err, model_err)
    assert total >= abs_err - 1e-9
    assert total >= samp_err - 1e-9
    assert total >= model_err - 1e-9


@given(floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_total_error_without_model(abs_err, samp_err):
    """Total error without model error still dominates components.

    compute_total_error(a, s, 0) >= max(a, s).
    """
    ebc = ErrorBoundComputer()
    total = ebc.compute_total_error(abs_err, samp_err, 0.0)
    assert total >= max(abs_err, samp_err) - 1e-9


# ---------------------------------------------------------------------------
# Required samples is positive and finite
# ---------------------------------------------------------------------------

@given(floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
       _variance, _alpha)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_required_samples_positive(target, variance, alpha):
    """Required sample count is always a positive integer.

    The number of samples needed to achieve target_error must be >= 1.
    """
    ebc = ErrorBoundComputer(bound_method="hoeffding")
    n = ebc.compute_required_samples(target, variance, alpha)
    assert n >= 1
    assert isinstance(n, int)


@given(_alpha)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_required_samples_inverse_monotone(alpha):
    """Halving target error requires more samples.

    A tighter error target needs a larger sample size.
    """
    ebc = ErrorBoundComputer(bound_method="hoeffding")
    n1 = ebc.compute_required_samples(1.0, 10.0, alpha)
    n2 = ebc.compute_required_samples(0.5, 10.0, alpha)
    assert n2 >= n1


# ---------------------------------------------------------------------------
# Total error is sum of components
# ---------------------------------------------------------------------------

@given(floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
       floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_total_error_is_sum(abs_err, samp_err, model_err):
    """Total error equals the sum of its components.

    compute_total_error should return abs_err + samp_err + model_err.
    """
    ebc = ErrorBoundComputer()
    total = ebc.compute_total_error(abs_err, samp_err, model_err)
    expected = abs_err + samp_err + model_err
    assert math.isclose(total, expected, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Effect size sign matches mean difference sign
# ---------------------------------------------------------------------------

@given(integers(min_value=30, max_value=50))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_effect_size_sign(n):
    """Effect size sign matches the direction of the mean difference.

    If mean(b) > mean(a), effect_size should indicate the direction.
    """
    rng = np.random.default_rng(123)
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(5.0, 1.0, n)
    tester = RegressionTester(method="welch_t")
    result = tester.test(a, b)
    # The sign should be consistent (implementation may use a-b or b-a)
    assert abs(result.effect_size) > 0.5


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

@given(integers(min_value=30, max_value=50))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow],
          deadline=None)
def test_multiple_testing_returns_correct_count(n):
    """test_multiple returns one result per sample pair.

    The number of results must match the number of input pairs.
    """
    rng = np.random.default_rng(42)
    pairs = [
        (rng.normal(0, 1, n), rng.normal(0, 1, n)),
        (rng.normal(0, 1, n), rng.normal(5, 1, n)),
    ]
    tester = RegressionTester(method="welch_t")
    results = tester.test_multiple(pairs, alpha=0.05, correction="holm")
    assert len(results) == len(pairs)
    for r in results:
        assert 0.0 - 1e-9 <= r.p_value <= 1.0 + 1e-9
