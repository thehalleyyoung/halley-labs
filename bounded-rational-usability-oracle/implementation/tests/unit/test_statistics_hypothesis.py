"""Unit tests for usability_oracle.statistics.hypothesis_tests.

Tests paired t-test, Wilcoxon signed-rank, permutation test, and bootstrap
test with known data and properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.statistics.hypothesis_tests import (
    BootstrapTest,
    PairedTTest,
    PermutationTest,
    WilcoxonSignedRank,
)
from usability_oracle.statistics.types import (
    AlternativeHypothesis,
    HypothesisTestResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ===================================================================
# PairedTTest
# ===================================================================


class TestPairedTTest:

    def test_reject_when_means_differ(self):
        rng = _rng()
        a = rng.normal(0.0, 1.0, 100)
        b = rng.normal(2.0, 1.0, 100)
        result = PairedTTest().test(a, b, alpha=0.05)
        assert isinstance(result, HypothesisTestResult)
        assert result.p_value < 0.05

    def test_no_reject_identical_samples(self):
        rng = _rng()
        a = rng.normal(5.0, 1.0, 50)
        result = PairedTTest().test(a, a, alpha=0.05)
        # With identical arrays, p-value is NaN or reject_null is False
        assert not result.reject_null

    def test_p_value_in_unit_interval(self):
        rng = _rng(99)
        a = rng.normal(0, 1, 30)
        b = rng.normal(0.5, 1, 30)
        result = PairedTTest().test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_two_sided_default(self):
        rng = _rng()
        a = rng.normal(0, 1, 40)
        b = rng.normal(1, 1, 40)
        result = PairedTTest().test(a, b)
        assert result.p_value < 0.05

    def test_greater_alternative(self):
        rng = _rng()
        a = rng.normal(0, 0.5, 50)
        b = rng.normal(1, 0.5, 50)
        # The paired t-test tests H1: b > a (second > first)
        result = PairedTTest().test(a, b, alternative=AlternativeHypothesis.GREATER)
        assert result.reject_null

    def test_less_alternative(self):
        rng = _rng()
        a = rng.normal(1, 0.5, 50)
        b = rng.normal(0, 0.5, 50)
        # The paired t-test tests H1: b < a (second < first)
        result = PairedTTest().test(a, b, alternative=AlternativeHypothesis.LESS)
        assert result.reject_null

    def test_effect_size_present(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = PairedTTest().test(a, b)
        assert result.effect_size is not None


# ===================================================================
# WilcoxonSignedRank
# ===================================================================


class TestWilcoxonSignedRank:

    def test_known_data_rejects(self):
        rng = _rng()
        a = rng.normal(0, 1, 50)
        b = rng.normal(2, 1, 50)
        result = WilcoxonSignedRank().test(a, b, alpha=0.05)
        assert isinstance(result, HypothesisTestResult)
        assert result.p_value < 0.05

    def test_identical_samples_not_rejected(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        result = WilcoxonSignedRank().test(a, a, alpha=0.05)
        # p-value should be high or NaN for zero differences
        assert result.p_value >= 0.05 or np.isnan(result.p_value)

    def test_p_value_bounded(self):
        rng = _rng(7)
        a = rng.normal(0, 1, 40)
        b = rng.normal(0.5, 1, 40)
        result = WilcoxonSignedRank().test(a, b)
        assert 0.0 <= result.p_value <= 1.0


# ===================================================================
# PermutationTest
# ===================================================================


class TestPermutationTest:

    def test_p_value_in_unit_interval(self):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = PermutationTest(n_permutations=1000).test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_detects_large_difference(self):
        rng = _rng()
        a = rng.normal(0, 0.5, 50)
        b = rng.normal(5, 0.5, 50)
        result = PermutationTest(n_permutations=1000).test(a, b, alpha=0.05)
        assert result.p_value < 0.05

    def test_no_difference_high_p_value(self):
        rng = _rng(123)
        a = rng.normal(0, 1, 40)
        # Same distribution
        b = rng.normal(0, 1, 40)
        result = PermutationTest(n_permutations=1000).test(a, b, alpha=0.05)
        assert result.p_value > 0.01  # should generally not reject


# ===================================================================
# BootstrapTest
# ===================================================================


class TestBootstrapTest:

    def test_p_value_in_unit_interval(self):
        rng = _rng()
        a = rng.normal(0, 1, 40)
        b = rng.normal(0.5, 1, 40)
        result = BootstrapTest(n_bootstrap=500).test(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_large_effect_detected(self):
        rng = _rng()
        a = rng.normal(0, 0.5, 60)
        b = rng.normal(3, 0.5, 60)
        result = BootstrapTest(n_bootstrap=500).test(a, b, alpha=0.05)
        assert result.p_value < 0.05


# ===================================================================
# Alternative hypothesis
# ===================================================================


class TestAlternatives:

    @pytest.mark.parametrize("alt", [
        AlternativeHypothesis.TWO_SIDED,
        AlternativeHypothesis.GREATER,
        AlternativeHypothesis.LESS,
    ])
    def test_all_alternatives_produce_valid_result(self, alt):
        rng = _rng()
        a = rng.normal(0, 1, 30)
        b = rng.normal(1, 1, 30)
        result = PairedTTest().test(a, b, alternative=alt)
        assert 0.0 <= result.p_value <= 1.0

    def test_one_sided_less_than_two_sided(self):
        rng = _rng()
        a = rng.normal(0, 1, 50)
        b = rng.normal(1, 1, 50)
        two_sided = PairedTTest().test(a, b, alternative=AlternativeHypothesis.TWO_SIDED)
        # b > a, so GREATER is the correct one-sided direction
        one_sided = PairedTTest().test(a, b, alternative=AlternativeHypothesis.GREATER)
        # In the correct direction, one-sided p ≤ two-sided p
        assert one_sided.p_value <= two_sided.p_value + 1e-10


# ===================================================================
# Type I error rate
# ===================================================================


class TestTypeIError:

    def test_type_i_error_rate_near_alpha(self):
        """Under H₀ (equal means), rejection rate should be near α."""
        alpha = 0.05
        n_simulations = 200
        rng = _rng(0)
        rejections = 0
        for _ in range(n_simulations):
            a = rng.normal(0, 1, 30)
            b = rng.normal(0, 1, 30)
            result = PairedTTest().test(a, b, alpha=alpha)
            if result.p_value < alpha:
                rejections += 1
        rate = rejections / n_simulations
        # Should be roughly α ± some tolerance
        assert 0.01 <= rate <= 0.15
