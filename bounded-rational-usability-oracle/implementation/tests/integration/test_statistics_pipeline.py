"""Integration test: statistical comparison pipeline.

Generates cost samples from two UI versions, runs hypothesis tests,
applies FDR correction, computes effect sizes, and verifies power analysis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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
    BootstrapCI,
    compute_power,
    required_sample_size,
    AlternativeHypothesis,
    EffectSizeType,
    CorrectionMethod,
)

# ---------------------------------------------------------------------------
# Helpers — Generate cost samples
# ---------------------------------------------------------------------------


def _generate_cost_samples(
    n: int = 50,
    mean_a: float = 5.0,
    mean_b: float = 5.0,
    std: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired cost samples from two UI versions."""
    rng = np.random.default_rng(seed)
    a = rng.normal(mean_a, std, size=n)
    b = rng.normal(mean_b, std, size=n)
    return a, b


# ===================================================================
# Tests — Large difference is detected
# ===================================================================


class TestLargeDifferenceDetection:
    """Generate samples with a large mean difference and verify detection."""

    def test_large_difference_detected_by_ttest(self):
        """Paired t-test detects a large mean difference."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=8.0, std=1.0)
        result = PairedTTest().test(a, b)
        assert result.reject_null, \
            f"Should reject null for large diff, p={result.p_value}"

    def test_large_difference_detected_by_wilcoxon(self):
        """Wilcoxon signed-rank detects a large difference."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=8.0, std=1.0)
        result = WilcoxonSignedRank().test(a, b)
        assert result.reject_null, \
            f"Should reject null for large diff, p={result.p_value}"

    def test_large_difference_detected_by_permutation(self):
        """Permutation test detects a large difference."""
        a, b = _generate_cost_samples(n=30, mean_a=5.0, mean_b=8.0, std=1.0)
        result = PermutationTest(n_permutations=5000).test(a, b)
        assert result.reject_null, \
            f"Should reject null for large diff, p={result.p_value}"

    def test_large_effect_size(self):
        """Large difference yields large effect size."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=8.0, std=1.0)
        es = cohens_d(a, b)
        assert abs(es.value) > 1.0, \
            f"Effect size should be > 1.0 for 3σ difference, got {es.value}"


# ===================================================================
# Tests — No difference is not detected
# ===================================================================


class TestNoDifferenceNotDetected:
    """Generate samples with no mean difference and verify non-rejection."""

    def test_no_difference_not_rejected_ttest(self):
        """No real difference → t-test should not reject (mostly)."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=5.0, std=1.0)
        result = PairedTTest().test(a, b)
        # With α=0.05, false positive rate should be ≤ 5%
        # One run may reject, but p should generally be > 0.01
        assert result.p_value > 0.01 or not result.reject_null, \
            f"Should generally not reject for no diff, p={result.p_value}"

    def test_no_difference_small_effect_size(self):
        """No real difference → small effect size."""
        a, b = _generate_cost_samples(n=100, mean_a=5.0, mean_b=5.0, std=1.0)
        es = cohens_d(a, b)
        assert abs(es.value) < 0.5, \
            f"No-difference effect size should be small, got {es.value}"

    def test_no_difference_ci_contains_zero(self):
        """No real difference → CI for mean difference contains zero."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=5.0, std=1.0)
        result = PairedTTest().test(a, b)
        ci = result.ci
        assert ci.lower <= 0 <= ci.upper or abs(ci.lower) < 1.0, \
            f"CI should contain 0 or be small: [{ci.lower}, {ci.upper}]"


# ===================================================================
# Tests — FDR control
# ===================================================================


class TestFDRControl:
    """Test that FDR correction controls false discoveries."""

    def test_fdr_controls_false_discoveries(self):
        """Under null hypothesis, BH controls FDR at α level."""
        rng = np.random.default_rng(42)
        n_tests = 20
        n_experiments = 200
        alpha = 0.05
        false_discovery_rates = []

        for _ in range(n_experiments):
            # All tests under null → all p-values should be uniform(0,1)
            p_values = rng.uniform(0, 1, size=n_tests)
            result = BenjaminiHochberg().correct(list(p_values), alpha=alpha)
            n_rejections = sum(result.rejected)
            # Under null, ALL rejections are false discoveries
            fdr = n_rejections / max(n_rejections, 1)
            false_discovery_rates.append(fdr if n_rejections > 0 else 0.0)

        mean_fdr = np.mean(false_discovery_rates)
        # BH controls E[FDR] ≤ α
        assert mean_fdr <= alpha + 0.05, \
            f"Mean FDR {mean_fdr} exceeds α={alpha} (with tolerance)"

    def test_bonferroni_more_conservative_than_bh(self):
        """Bonferroni should reject ≤ BH rejections for same p-values."""
        rng = np.random.default_rng(42)
        # Mix of significant and non-significant p-values
        p_values = list(rng.uniform(0, 0.1, size=5)) + \
                   list(rng.uniform(0.1, 1.0, size=15))
        bonf = BonferroniCorrection().correct(p_values, alpha=0.05)
        bh = BenjaminiHochberg().correct(p_values, alpha=0.05)
        assert sum(bonf.rejected) <= sum(bh.rejected), \
            f"Bonferroni ({sum(bonf.rejected)}) should reject ≤ BH ({sum(bh.rejected)})"

    def test_holm_between_bonferroni_and_bh(self):
        """Holm rejects ≥ Bonferroni (same FWER, more power)."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.06, 0.10, 0.20, 0.50]
        bonf = BonferroniCorrection().correct(p_values, alpha=0.05)
        holm = HolmBonferroni().correct(p_values, alpha=0.05)
        assert sum(holm.rejected) >= sum(bonf.rejected), \
            f"Holm ({sum(holm.rejected)}) should reject ≥ Bonferroni ({sum(bonf.rejected)})"

    def test_multiple_correction_preserves_order(self):
        """Adjusted p-value ordering should match original ordering."""
        p_values = [0.01, 0.05, 0.10, 0.50]
        result = BenjaminiHochberg().correct(p_values, alpha=0.05)
        adj = list(result.adjusted_p_values)
        # Adjusted values should be monotonically non-decreasing
        # when original p-values are already sorted
        for i in range(1, len(adj)):
            assert adj[i] >= adj[i - 1] - 1e-10


# ===================================================================
# Tests — Power analysis
# ===================================================================


class TestPowerAnalysisPipeline:
    """Power analysis predicts required sample size."""

    def test_power_analysis_predicts_sample_size(self):
        """Required sample size achieves target power."""
        d = 0.5  # medium effect
        target_power = 0.80
        result = required_sample_size(d, power=target_power, alpha=0.05)
        # Verify the predicted n actually achieves the target power
        actual_power = compute_power(d, result.required_sample_size, alpha=0.05)
        assert actual_power >= target_power - 0.01, \
            f"Predicted n={result.required_sample_size} gives power={actual_power}"

    def test_power_analysis_small_effect_needs_more_samples(self):
        """Small effect sizes require more samples than large effects."""
        n_small = required_sample_size(0.2, power=0.80, alpha=0.05)
        n_large = required_sample_size(0.8, power=0.80, alpha=0.05)
        assert n_small.required_sample_size > n_large.required_sample_size

    def test_power_analysis_higher_power_needs_more_samples(self):
        """Higher power targets require more samples."""
        n_80 = required_sample_size(0.5, power=0.80, alpha=0.05)
        n_95 = required_sample_size(0.5, power=0.95, alpha=0.05)
        assert n_95.required_sample_size > n_80.required_sample_size

    def test_observed_rejection_matches_predicted_power(self):
        """Simulated rejection rate approximates predicted power."""
        d = 0.8  # use larger effect for clearer signal
        n = 30
        alpha = 0.05
        predicted_power = compute_power(d, n, alpha=alpha)

        rng = np.random.default_rng(42)
        n_simulations = 1000
        rejections = 0
        for _ in range(n_simulations):
            a = rng.normal(0, 1, size=n)
            b = rng.normal(d, 1, size=n)
            result = PairedTTest().test(a, b, alpha=alpha)
            if result.reject_null:
                rejections += 1

        observed_power = rejections / n_simulations
        assert abs(observed_power - predicted_power) < 0.20, \
            f"Observed power {observed_power} far from predicted {predicted_power}"


# ===================================================================
# Tests — Bootstrap CI pipeline
# ===================================================================


class TestBootstrapCIPipeline:
    """Bootstrap CI integration with comparison pipeline."""

    def test_bootstrap_ci_covers_true_mean(self):
        """Bootstrap CI should cover the true mean (most of the time)."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        data = rng.normal(true_mean, 1.0, size=100)
        bc = BootstrapCI(n_bootstrap=2000, seed=42)
        result = bc.percentile_ci(data, np.mean, alpha=0.05)
        assert result.ci.lower <= true_mean <= result.ci.upper, \
            f"CI [{result.ci.lower}, {result.ci.upper}] misses μ={true_mean}"

    def test_bootstrap_ci_width_reasonable(self):
        """Bootstrap CI width should be reasonable for n=100."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, size=100)
        bc = BootstrapCI(n_bootstrap=2000, seed=42)
        result = bc.percentile_ci(data, np.mean, alpha=0.05)
        width = result.ci.width
        # For n=100, σ=1, 95% CI width ≈ 4 * SE ≈ 4 * 0.1 = 0.4
        assert width < 1.0, f"CI too wide: {width}"
        assert width > 0.01, f"CI too narrow: {width}"


# ===================================================================
# Tests — Full comparison chain
# ===================================================================


class TestFullComparisonChain:
    """End-to-end: samples → test → correction → effect size → power."""

    def test_full_chain_large_difference(self):
        """Full pipeline correctly identifies large regression."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=8.0, std=1.0)

        # Step 1: Hypothesis test
        test_result = PairedTTest().test(a, b)
        assert test_result.reject_null

        # Step 2: Effect size
        es = cohens_d(a, b)
        assert abs(es.value) > 1.0

        # Step 3: Multiple correction (single test = trivial)
        correction = BenjaminiHochberg().correct(
            [test_result.p_value], alpha=0.05
        )
        assert correction.num_rejections == 1

    def test_full_chain_no_difference(self):
        """Full pipeline correctly identifies no regression."""
        a, b = _generate_cost_samples(n=50, mean_a=5.0, mean_b=5.0, std=1.0)

        test_result = PairedTTest().test(a, b)
        es = cohens_d(a, b)

        # Should not detect meaningful difference
        assert abs(es.value) < 0.5 or not test_result.reject_null

    def test_multiple_metrics_fdr_control(self):
        """Test FDR control across multiple metric comparisons."""
        rng = np.random.default_rng(42)
        n = 30
        n_metrics = 10
        p_values = []

        for i in range(n_metrics):
            a = rng.normal(5.0, 1.0, size=n)
            # 3 metrics have real differences, 7 do not
            if i < 3:
                b = rng.normal(7.0, 1.0, size=n)
            else:
                b = rng.normal(5.0, 1.0, size=n)
            result = PairedTTest().test(a, b)
            p_values.append(result.p_value)

        correction = BenjaminiHochberg().correct(p_values, alpha=0.05)
        # Should detect the 3 real differences
        assert correction.num_rejections >= 2, \
            f"Should detect most real differences, got {correction.num_rejections}"
