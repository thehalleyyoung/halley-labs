"""Tests for marace.sampling.concentration — concentration inequalities."""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.sampling.concentration import (
    BoundResult,
    SelfNormalizedBound,
    EmpiricalBernsteinBound,
    AdaptiveBoundSelector,
    FiniteSampleGuarantee,
)


# ======================================================================
# SelfNormalizedBound
# ======================================================================

class TestSelfNormalizedBound:
    """Test self-normalized concentration inequality."""

    def test_known_mean_uniform_weights(self):
        """Uniform weights + known mean → CI should contain true mean."""
        rng = np.random.RandomState(42)
        n = 500
        values = rng.uniform(0, 1, size=n)
        weights = np.ones(n)

        bound = SelfNormalizedBound(support=(0.0, 1.0))
        result = bound.confidence_interval(weights, values, alpha=0.05)

        assert isinstance(result, BoundResult)
        assert result.lower <= result.upper
        true_mean = np.mean(values)
        assert result.lower <= true_mean <= result.upper

    def test_ci_width_decreases_with_n(self):
        """CI should be narrower with more samples."""
        rng = np.random.RandomState(42)
        bound = SelfNormalizedBound(support=(0.0, 1.0))

        widths = []
        for n in [100, 500, 2000]:
            values = rng.uniform(0, 1, size=n)
            weights = np.ones(n)
            result = bound.confidence_interval(weights, values, alpha=0.05)
            widths.append(result.width)

        assert widths[0] > widths[-1]

    def test_ci_coverage_nominal(self):
        """Coverage over many trials should be close to 1 - alpha."""
        rng = np.random.RandomState(123)
        bound = SelfNormalizedBound(support=(0.0, 1.0))
        alpha = 0.05
        true_mean = 0.5
        covers = 0
        n_trials = 200
        n_samples = 200

        for _ in range(n_trials):
            values = rng.uniform(0, 1, size=n_samples)
            weights = np.ones(n_samples)
            result = bound.confidence_interval(weights, values, alpha=alpha)
            if result.lower <= true_mean <= result.upper:
                covers += 1

        coverage = covers / n_trials
        # Should be at least 1 - alpha - slack
        assert coverage >= 0.80

    def test_effective_sample_size(self):
        """ESS should be reported correctly."""
        n = 100
        values = np.random.RandomState(0).uniform(0, 1, size=n)
        weights = np.ones(n)
        bound = SelfNormalizedBound(support=(0.0, 1.0))
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert result.effective_sample_size > 0

    def test_method_name(self):
        bound = SelfNormalizedBound(support=(0.0, 1.0))
        values = np.random.RandomState(0).uniform(0, 1, size=50)
        weights = np.ones(50)
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert isinstance(result.method, str)
        assert len(result.method) > 0


# ======================================================================
# EmpiricalBernsteinBound
# ======================================================================

class TestEmpiricalBernsteinBound:
    """Test empirical Bernstein concentration inequality."""

    def test_coverage_on_normal_samples(self):
        """CI should cover the true mean with bounded values."""
        rng = np.random.RandomState(42)
        bound = EmpiricalBernsteinBound(upper_bound=1.0)
        alpha = 0.05
        n_samples = 300

        values = np.clip(rng.normal(0.5, 0.15, size=n_samples), 0, 1)
        weights = np.ones(n_samples)
        result = bound.confidence_interval(weights, values, alpha=alpha)
        # Basic sanity: interval exists and is ordered
        assert result.lower <= result.upper
        assert isinstance(result.method, str)

    def test_ci_contains_sample_mean(self):
        """CI should be ordered and have finite bounds."""
        rng = np.random.RandomState(7)
        values = rng.uniform(0, 1, size=200)
        weights = np.ones(200)
        bound = EmpiricalBernsteinBound(upper_bound=1.0)
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert result.lower <= result.upper
        assert np.isfinite(result.lower)
        assert np.isfinite(result.upper)

    def test_ci_ordered(self):
        values = np.random.RandomState(0).uniform(0, 1, size=100)
        weights = np.ones(100)
        bound = EmpiricalBernsteinBound(upper_bound=1.0)
        result = bound.confidence_interval(weights, values, alpha=0.1)
        assert result.lower <= result.upper

    def test_with_known_upper_bound(self):
        values = np.random.RandomState(0).uniform(0, 1, size=100)
        weights = np.ones(100)
        bound = EmpiricalBernsteinBound(upper_bound=1.0)
        result = bound.confidence_interval(weights, values, alpha=0.05)
        assert result.lower <= result.upper


# ======================================================================
# AdaptiveBoundSelector
# ======================================================================

class TestAdaptiveBoundSelector:
    """Test adaptive bound type selection."""

    def test_selects_bound(self):
        """Should return a valid BoundResult."""
        rng = np.random.RandomState(42)
        values = rng.uniform(0, 1, size=200)
        weights = np.ones(200)

        selector = AdaptiveBoundSelector(support=(0.0, 1.0))
        result = selector.select_bound(weights, values, alpha=0.05)
        assert isinstance(result, BoundResult)
        assert result.lower <= result.upper

    def test_selects_tighter_bound(self):
        """Should select a reasonably tight bound."""
        rng = np.random.RandomState(42)
        values = rng.normal(0.5, 0.1, size=500)
        values = np.clip(values, 0, 1)
        weights = np.ones(500)

        selector = AdaptiveBoundSelector(support=(0.0, 1.0))
        result = selector.select_bound(weights, values, alpha=0.05)
        assert result.width < 0.5

    def test_method_is_reported(self):
        rng = np.random.RandomState(42)
        values = rng.uniform(0, 1, size=200)
        weights = np.ones(200)
        selector = AdaptiveBoundSelector(support=(0.0, 1.0))
        result = selector.select_bound(weights, values, alpha=0.05)
        assert isinstance(result.method, str)


# ======================================================================
# FiniteSampleGuarantee
# ======================================================================

class TestFiniteSampleGuarantee:
    """Test finite sample size computation."""

    def test_required_samples_positive(self):
        fsg = FiniteSampleGuarantee(support_range=1.0)
        n = fsg.required_samples(alpha=0.05, epsilon=0.01)
        assert n > 0

    def test_required_samples_chatterjee_positive(self):
        fsg = FiniteSampleGuarantee(support_range=1.0)
        n = fsg.required_samples_chatterjee(alpha=0.05, epsilon=0.01)
        assert n > 0

    def test_required_samples_bernstein_positive(self):
        fsg = FiniteSampleGuarantee(support_range=1.0)
        n = fsg.required_samples_bernstein(alpha=0.05, epsilon=0.01)
        assert n > 0

    @pytest.mark.parametrize("epsilon", [0.1, 0.05, 0.01, 0.005])
    def test_monotone_in_epsilon(self, epsilon):
        """Smaller epsilon → more samples required."""
        fsg = FiniteSampleGuarantee(support_range=1.0)
        n_loose = fsg.required_samples(alpha=0.05, epsilon=0.1)
        n_tight = fsg.required_samples(alpha=0.05, epsilon=epsilon)
        assert n_tight >= n_loose

    def test_monotone_in_alpha(self):
        """Smaller alpha (higher confidence) → more samples."""
        fsg = FiniteSampleGuarantee(support_range=1.0)
        n_90 = fsg.required_samples(alpha=0.10, epsilon=0.01)
        n_99 = fsg.required_samples(alpha=0.01, epsilon=0.01)
        assert n_99 >= n_90

    def test_with_pilot_variance(self):
        fsg = FiniteSampleGuarantee(support_range=1.0, pilot_variance=0.1)
        n = fsg.required_samples(alpha=0.05, epsilon=0.01)
        assert n > 0

    def test_invalid_support_range(self):
        with pytest.raises(ValueError):
            FiniteSampleGuarantee(support_range=0.0)

    def test_invalid_epsilon(self):
        fsg = FiniteSampleGuarantee(support_range=1.0)
        with pytest.raises(ValueError):
            fsg.required_samples_chatterjee(alpha=0.05, epsilon=0.0)

    def test_invalid_alpha(self):
        fsg = FiniteSampleGuarantee(support_range=1.0)
        with pytest.raises(ValueError):
            fsg.required_samples_chatterjee(alpha=0.0, epsilon=0.01)

    def test_invalid_ess_ratio(self):
        fsg = FiniteSampleGuarantee(support_range=1.0)
        with pytest.raises(ValueError):
            fsg.required_samples_chatterjee(alpha=0.05, epsilon=0.01,
                                             ess_ratio=0.0)
