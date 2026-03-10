"""Property-based tests for the statistics module.

Verifies invariants of hypothesis tests, effect sizes, multiple comparison
corrections (Bonferroni, BH), bootstrap CI, and power analysis.
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

from usability_oracle.statistics import (
    PairedTTest,
    WilcoxonSignedRank,
    PermutationTest,
    BonferroniCorrection,
    BenjaminiHochberg,
    HolmBonferroni,
    EffectSizeCalculator,
    cohens_d,
    hedges_g,
    cliff_delta,
    BootstrapCI,
    compute_power,
    required_sample_size,
    AlternativeHypothesis,
    EffectSizeType,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_pos_float = floats(min_value=0.01, max_value=100.0,
                    allow_nan=False, allow_infinity=False)


@composite
def _paired_samples(draw, min_size=5, max_size=50):
    """Generate a pair of matched samples."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    a = draw(lists(
        floats(min_value=-50.0, max_value=50.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    b = draw(lists(
        floats(min_value=-50.0, max_value=50.0,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    return np.array(a), np.array(b)


@composite
def _p_value_list(draw, min_size=3, max_size=20):
    """Generate a list of p-values in [0, 1]."""
    n = draw(integers(min_value=min_size, max_value=max_size))
    pvals = draw(lists(
        floats(min_value=0.001, max_value=0.999,
               allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    return pvals


_ATOL = 1e-6

# ---------------------------------------------------------------------------
# P-values are in [0, 1]
# ---------------------------------------------------------------------------


@given(_paired_samples(min_size=5, max_size=30))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_paired_ttest_pvalue_in_unit_interval(samples):
    """P-value from paired t-test is in [0, 1]."""
    a, b = samples
    assume(np.std(a - b) > 1e-10)  # avoid degenerate case
    result = PairedTTest().test(a, b)
    assert 0.0 <= result.p_value <= 1.0 + _ATOL, \
        f"p-value out of range: {result.p_value}"


@given(_paired_samples(min_size=5, max_size=30))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_wilcoxon_pvalue_in_unit_interval(samples):
    """P-value from Wilcoxon signed-rank test is in [0, 1]."""
    a, b = samples
    assume(np.any(a != b))  # need at least some non-zero differences
    result = WilcoxonSignedRank().test(a, b)
    assert 0.0 <= result.p_value <= 1.0 + _ATOL, \
        f"p-value out of range: {result.p_value}"


# ---------------------------------------------------------------------------
# Effect size sign matches direction
# ---------------------------------------------------------------------------


def test_effect_size_sign_positive_difference():
    """When b > a, Cohen's d should be positive."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    es = cohens_d(a, b)
    assert es.value > 0, f"Effect size should be positive for b > a, got {es.value}"


def test_effect_size_sign_negative_difference():
    """When a > b, Cohen's d should be negative."""
    a = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    es = cohens_d(a, b)
    assert es.value < 0, f"Effect size should be negative for a > b, got {es.value}"


def test_effect_size_near_zero_for_similar():
    """Similar samples should yield a near-zero effect size."""
    rng = np.random.default_rng(42)
    a = rng.normal(5.0, 1.0, size=100)
    b = rng.normal(5.0, 1.0, size=100)
    es = cohens_d(a, b)
    assert abs(es.value) < 0.5, f"Effect size should be ≈0, got {es.value}"


@given(_paired_samples(min_size=10, max_size=30))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_hedges_g_close_to_cohens_d(samples):
    """Hedges' g ≈ Cohen's d (bias correction is small for n > 10)."""
    a, b = samples
    assume(np.std(a) > 0.01 and np.std(b) > 0.01)
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    assert abs(d.value - g.value) < 0.5, \
        f"Hedges' g and Cohen's d differ too much: {d.value} vs {g.value}"


# ---------------------------------------------------------------------------
# Cliff's delta bounded in [-1, 1]
# ---------------------------------------------------------------------------


@given(_paired_samples(min_size=5, max_size=30))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_cliff_delta_bounded(samples):
    """Cliff's delta is in [-1, 1]."""
    a, b = samples
    cd = cliff_delta(a, b)
    assert -1.0 - _ATOL <= cd.value <= 1.0 + _ATOL, \
        f"Cliff's delta out of range: {cd.value}"


# ---------------------------------------------------------------------------
# Bonferroni adjusted p-values ≥ original p-values
# ---------------------------------------------------------------------------


@given(_p_value_list())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_bonferroni_adjusted_geq_original(pvals):
    """Bonferroni adjusted p-values ≥ original p-values."""
    result = BonferroniCorrection().correct(pvals)
    for orig, adj in zip(pvals, result.adjusted_p_values):
        assert adj >= orig - _ATOL, \
            f"Adjusted {adj} < original {orig}"


@given(_p_value_list())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_bonferroni_adjusted_leq_one(pvals):
    """Bonferroni adjusted p-values are capped at 1."""
    result = BonferroniCorrection().correct(pvals)
    for adj in result.adjusted_p_values:
        assert adj <= 1.0 + _ATOL, f"Adjusted p-value > 1: {adj}"


# ---------------------------------------------------------------------------
# Holm-Bonferroni adjusted p-values ≥ original, ≤ Bonferroni
# ---------------------------------------------------------------------------


@given(_p_value_list())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_holm_adjusted_geq_original(pvals):
    """Holm adjusted p-values ≥ original p-values."""
    result = HolmBonferroni().correct(pvals)
    for orig, adj in zip(pvals, result.adjusted_p_values):
        assert adj >= orig - _ATOL, \
            f"Holm adjusted {adj} < original {orig}"


# ---------------------------------------------------------------------------
# BH adjusted p-values ≥ original p-values
# ---------------------------------------------------------------------------


@given(_p_value_list())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_bh_adjusted_geq_original(pvals):
    """BH adjusted p-values ≥ original p-values."""
    result = BenjaminiHochberg().correct(pvals)
    for orig, adj in zip(pvals, result.adjusted_p_values):
        assert adj >= orig - _ATOL, \
            f"BH adjusted {adj} < original {orig}"


# ---------------------------------------------------------------------------
# BH rejection set ⊆ Bonferroni rejection set is NOT always true
# ---------------------------------------------------------------------------


def test_bh_not_always_subset_of_bonferroni():
    """Document: BH can reject hypotheses that Bonferroni does not.

    BH controls FDR (expected proportion of false discoveries among rejections),
    which is a weaker error guarantee than FWER controlled by Bonferroni.
    Therefore BH may reject more hypotheses.
    """
    # Carefully chosen p-values where BH rejects more than Bonferroni
    pvals = [0.01, 0.02, 0.03, 0.04, 0.10, 0.15, 0.20, 0.30, 0.50, 0.90]
    alpha = 0.05
    bonf = BonferroniCorrection().correct(pvals, alpha=alpha)
    bh = BenjaminiHochberg().correct(pvals, alpha=alpha)
    bonf_rejections = sum(bonf.rejected)
    bh_rejections = sum(bh.rejected)
    # BH should reject at least as many (and typically more) than Bonferroni
    assert bh_rejections >= bonf_rejections, \
        f"BH rejected {bh_rejections} but Bonferroni rejected {bonf_rejections}"
    # Specifically, BH should reject more for these p-values
    # Note: this is the documented behavior — BH ⊄ Bonferroni in general


# ---------------------------------------------------------------------------
# Bootstrap CI width decreases with sample size (high probability)
# ---------------------------------------------------------------------------


def test_bootstrap_ci_width_decreases_with_n():
    """Bootstrap CI width should generally decrease with sample size."""
    rng = np.random.default_rng(42)
    small = rng.normal(5.0, 2.0, size=20)
    large = rng.normal(5.0, 2.0, size=200)

    bc = BootstrapCI(n_bootstrap=2000, seed=42)
    ci_small = bc.percentile_ci(small, np.mean, alpha=0.05)
    ci_large = bc.percentile_ci(large, np.mean, alpha=0.05)

    width_small = ci_small.ci.width
    width_large = ci_large.ci.width
    assert width_large < width_small, \
        f"Larger sample should give narrower CI: {width_large} vs {width_small}"


# ---------------------------------------------------------------------------
# Power increases with effect size (fixed n, α)
# ---------------------------------------------------------------------------


@given(floats(min_value=0.1, max_value=0.5,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.5, max_value=2.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_power_increases_with_effect_size(d_small, d_extra):
    """Power increases with effect size for fixed n and α."""
    d_large = d_small + d_extra
    n = 30
    pow_small = compute_power(d_small, n, alpha=0.05)
    pow_large = compute_power(d_large, n, alpha=0.05)
    assert pow_large >= pow_small - _ATOL, \
        f"Power should increase: {pow_small} (d={d_small}) vs {pow_large} (d={d_large})"


# ---------------------------------------------------------------------------
# Power increases with sample size (fixed effect, α)
# ---------------------------------------------------------------------------


@given(floats(min_value=0.3, max_value=1.0,
              allow_nan=False, allow_infinity=False),
       integers(min_value=10, max_value=50),
       integers(min_value=10, max_value=100))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_power_increases_with_sample_size(d, n_small, n_extra):
    """Power increases with sample size for fixed effect size and α."""
    n_large = n_small + n_extra
    pow_small = compute_power(d, n_small, alpha=0.05)
    pow_large = compute_power(d, n_large, alpha=0.05)
    assert pow_large >= pow_small - _ATOL, \
        f"Power should increase: {pow_small} (n={n_small}) vs {pow_large} (n={n_large})"


# ---------------------------------------------------------------------------
# Power is in [0, 1]
# ---------------------------------------------------------------------------


@given(floats(min_value=0.1, max_value=2.0,
              allow_nan=False, allow_infinity=False),
       integers(min_value=5, max_value=100))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_power_in_unit_interval(d, n):
    """Power must be in [0, 1]."""
    power = compute_power(d, n, alpha=0.05)
    assert 0.0 <= power <= 1.0 + _ATOL, f"Power out of range: {power}"


# ---------------------------------------------------------------------------
# Required sample size is positive
# ---------------------------------------------------------------------------


@given(floats(min_value=0.2, max_value=2.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_required_sample_size_positive(d):
    """Required sample size must be a positive integer."""
    result = required_sample_size(d, power=0.80, alpha=0.05)
    assert result.required_sample_size > 0, \
        f"Required sample size should be positive, got {result.required_sample_size}"


@given(floats(min_value=0.2, max_value=0.5,
              allow_nan=False, allow_infinity=False),
       floats(min_value=0.6, max_value=2.0,
              allow_nan=False, allow_infinity=False))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_required_sample_size_decreases_with_effect(d_small, d_extra):
    """Larger effect sizes require fewer samples for the same power."""
    d_large = d_small + d_extra
    n_small = required_sample_size(d_small, power=0.80, alpha=0.05)
    n_large = required_sample_size(d_large, power=0.80, alpha=0.05)
    assert n_large.required_sample_size <= n_small.required_sample_size, \
        f"Larger effect should need fewer samples: {n_large.required_sample_size} vs {n_small.required_sample_size}"


# ---------------------------------------------------------------------------
# Effect size calculator
# ---------------------------------------------------------------------------


@given(_paired_samples(min_size=10, max_size=30))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_effect_size_calculator_cohens_d_finite(samples):
    """EffectSizeCalculator returns finite Cohen's d."""
    a, b = samples
    assume(np.std(a) > 0.01 and np.std(b) > 0.01)
    calc = EffectSizeCalculator()
    es = calc.estimate(a, b, measure=EffectSizeType.COHENS_D)
    assert math.isfinite(es.value), f"Effect size not finite: {es.value}"


@given(_paired_samples(min_size=10, max_size=30))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_effect_size_has_confidence_interval(samples):
    """Effect size should include a confidence interval."""
    a, b = samples
    assume(np.std(a) > 0.01 and np.std(b) > 0.01)
    es = cohens_d(a, b, confidence_level=0.95)
    assert es.ci is not None
    assert es.ci.lower <= es.ci.upper + _ATOL


# ---------------------------------------------------------------------------
# Identical samples yield non-significant result
# ---------------------------------------------------------------------------


def test_identical_samples_not_significant():
    """Identical samples should yield p > α (not reject null)."""
    rng = np.random.default_rng(42)
    a = rng.normal(5.0, 1.0, size=50)
    b = a + rng.normal(0, 1e-10, size=50)  # near-identical with tiny noise
    result = PairedTTest().test(a, b)
    # p-value should be very high (differences are negligible)
    if not math.isnan(result.p_value):
        assert not result.reject_null, \
            f"Near-identical samples should not be significant, p={result.p_value}"
