"""Tests for the analysis module.

These tests verify the analysis pipeline including composite testing,
price analysis, correlation analysis, bootstrap methods, and hypothesis testing.
Since the analysis submodules are not yet fully implemented, tests import from
the top-level types and statistics utilities that do exist, plus stub the
analysis-specific imports where needed.
"""

import pytest
import numpy as np
import math

from collusion_proof.types import (
    Verdict,
    TestTier,
    NullHypothesis,
    HypothesisTestResult,
    TestResult,
    ConfidenceInterval,
    CollusionPremiumResult,
    GameConfig,
    MarketType,
    DemandSystem,
)
from collusion_proof.config import (
    TestConfig,
    MarketConfig,
    ALPHA_ALLOCATION,
    DEFAULT_ALPHA,
)
from collusion_proof.statistics_utils import (
    bootstrap_mean,
    bootstrap_statistic,
    bca_confidence_interval,
    block_bootstrap,
    permutation_test,
    holm_bonferroni,
    benjamini_hochberg,
    fisher_combine_pvalues,
    stouffer_combine_pvalues,
    harmonic_mean_pvalue,
    effect_size_cohens_d,
    effect_size_glass_delta,
    newey_west_se,
    kernel_density_estimate,
    empirical_cdf,
    ks_test_statistic,
    ljung_box_test,
    durbin_watson,
    weighted_mean,
    weighted_variance,
    trimmed_mean,
    winsorized_mean,
    robust_std,
)
from collusion_proof.interval_arithmetic import (
    Interval,
    interval_mean,
    interval_variance,
    interval_correlation,
    interval_collusion_premium,
)
from collusion_proof.utils import (
    moving_average,
    exponential_moving_average,
    detect_convergence,
    compute_nash_price,
    compute_monopoly_price,
    compute_demand,
    compute_profit,
    safe_divide,
    validate_price_matrix,
    ensure_2d,
    windowed_statistic,
)
from collusion_proof.cli.commands import run_analysis


class TestAlphaSpending:
    """Tests for alpha allocation across tiers."""

    def test_initial_allocation_sums_to_alpha(self):
        total = sum(ALPHA_ALLOCATION.values())
        assert abs(total - DEFAULT_ALPHA) < 1e-10

    def test_tier_budgets_positive(self):
        for tier, budget in ALPHA_ALLOCATION.items():
            assert budget > 0, f"{tier} has non-positive budget"

    def test_four_tiers_allocated(self):
        assert len(ALPHA_ALLOCATION) == 4

    def test_effective_alpha_known_tier(self):
        cfg = TestConfig()
        assert cfg.effective_alpha("tier1") == 0.02

    def test_effective_alpha_unknown_tier(self):
        cfg = TestConfig()
        result = cfg.effective_alpha("unknown_tier")
        assert result == cfg.alpha / len(cfg.alpha_allocation)


class TestAnalysisPipeline:
    """Tests for the run_analysis pipeline from CLI commands."""

    def test_competitive_prices_detected(self):
        rng = np.random.RandomState(42)
        prices = rng.normal(1.05, 0.1, (5000, 2))
        prices = np.clip(prices, 0, None)
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert result["verdict"] in ("competitive", "inconclusive")
        assert result["collusion_index"] < 0.3

    def test_collusive_prices_detected(self):
        rng = np.random.RandomState(42)
        prices = rng.normal(5.0, 0.1, (5000, 2))
        prices = np.clip(prices, 0, None)
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert result["verdict"] in ("collusive", "suspicious")
        assert result["collusion_index"] > 0.7

    def test_result_keys(self):
        prices = np.ones((100, 2)) * 3.0
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert "verdict" in result
        assert "confidence" in result
        assert "collusion_premium" in result
        assert "tier_results" in result
        assert isinstance(result["tier_results"], list)

    def test_collusion_premium_computation(self):
        prices = np.ones((100, 2)) * 3.25
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        expected_premium = (3.25 - 1.0) / (5.5 - 1.0)
        assert abs(result["collusion_premium"] - expected_premium) < 0.01

    def test_confidence_bounded(self):
        prices = np.ones((100, 2)) * 3.0
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert 0.0 <= result["confidence"] <= 1.0


class TestPriceAnalysis:
    """Tests for price-level statistical properties."""

    def test_supra_competitive_ratio(self):
        prices = np.ones((1000, 2)) * 3.0
        nash, monopoly = 1.0, 5.5
        premium = (np.mean(prices) - nash) / (monopoly - nash)
        assert 0.0 < premium < 1.0

    def test_price_dispersion_low_for_collusion(self):
        rng = np.random.RandomState(42)
        prices = rng.normal(5.0, 0.05, (10000, 2))
        dispersion = float(np.std(prices.mean(axis=1)))
        assert dispersion < 0.1

    def test_convergence_detection_stationary(self):
        rng = np.random.RandomState(42)
        data = np.concatenate([
            np.linspace(1.0, 5.0, 500),
            rng.normal(5.0, 0.001, 5000),
        ])
        idx = detect_convergence(data, window=100, threshold=0.01)
        assert idx is not None
        assert idx < 1000

    def test_convergence_detection_no_convergence(self):
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 10, 1000)
        idx = detect_convergence(data, window=100, threshold=0.001)
        assert idx is None

    def test_relative_price_level(self):
        nash = compute_nash_price(1.0, 10.0, 1.0, 2)
        monopoly = compute_monopoly_price(1.0, 10.0, 1.0)
        assert nash < monopoly
        assert nash >= 1.0  # at least marginal cost


class TestCorrelationAnalysis:
    """Tests for correlation analysis between players' prices."""

    def test_pearson_independent(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 10000)
        y = rng.normal(0, 1, 10000)
        r = float(np.corrcoef(x, y)[0, 1])
        assert abs(r) < 0.05

    def test_pearson_correlated(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 10000)
        y = x + rng.normal(0, 0.1, 10000)
        r = float(np.corrcoef(x, y)[0, 1])
        assert r > 0.9

    def test_interval_correlation_contains_sample(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 200)
        y = 0.5 * x + rng.normal(0, 1, 200)
        iv = interval_correlation(x, y)
        r_sample = float(np.corrcoef(x, y)[0, 1])
        assert iv.contains(r_sample)

    def test_interval_correlation_bounds(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        iv = interval_correlation(x, y)
        assert iv.lo >= -1.0
        assert iv.hi <= 1.0

    def test_granger_causality_proxy(self):
        """Test that lagged correlation can detect leader-follower patterns."""
        rng = np.random.RandomState(42)
        leader = rng.normal(5.0, 0.5, 1000)
        follower = np.roll(leader, 1) + rng.normal(0, 0.1, 1000)
        follower[0] = leader[0]
        lagged_corr = float(np.corrcoef(leader[:-1], follower[1:])[0, 1])
        assert lagged_corr > 0.7


class TestBootstrap:
    """Tests for bootstrap methods."""

    def test_nonparametric_mean(self):
        rng = np.random.RandomState(42)
        data = rng.normal(5.0, 1.0, 500)
        lo, pt, hi = bootstrap_mean(data, n_bootstrap=2000, random_state=42)
        assert lo < 5.0 < hi
        assert abs(pt - 5.0) < 0.2

    def test_bca_coverage(self):
        rng = np.random.RandomState(42)
        data = rng.normal(10.0, 2.0, 200)
        lo, pt, hi = bca_confidence_interval(
            data, statistic=np.mean, n_bootstrap=2000, confidence=0.95,
            random_state=42,
        )
        assert lo < 10.0 < hi

    def test_block_bootstrap_preserves_autocorrelation(self):
        rng = np.random.RandomState(42)
        data = np.cumsum(rng.normal(0, 1, 500))
        resamples = block_bootstrap(
            data, block_size=20, n_bootstrap=500,
            statistic=np.mean, random_state=42,
        )
        assert len(resamples) == 500
        assert not np.all(resamples == resamples[0])

    def test_bootstrap_statistic_custom(self):
        rng = np.random.RandomState(42)
        data = rng.exponential(2.0, 300)
        lo, pt, hi = bootstrap_statistic(
            data, statistic=np.median, n_bootstrap=2000,
            confidence=0.95, random_state=42,
        )
        true_median = np.log(2) * 2.0
        assert lo < true_median < hi


class TestHypothesisTesting:
    """Tests for hypothesis testing framework."""

    def test_t_test_reject(self):
        rng = np.random.RandomState(42)
        x = rng.normal(5.0, 1.0, 100)
        y = rng.normal(3.0, 1.0, 100)
        obs, p = permutation_test(x, y, statistic=lambda a, b: np.mean(a) - np.mean(b),
                                  n_permutations=2000, random_state=42)
        assert p < 0.05
        assert obs > 0

    def test_permutation_test_null(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        obs, p = permutation_test(x, y, statistic=lambda a, b: np.mean(a) - np.mean(b),
                                  n_permutations=2000, random_state=42)
        assert p > 0.05

    def test_holm_bonferroni_conservative(self):
        p_values = np.array([0.001, 0.01, 0.04, 0.5])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)
        # Holm is more conservative than unadjusted
        assert all(a >= o for a, o in zip(adjusted, p_values))
        assert reject[0]  # smallest p should still be rejected
        assert not reject[3]  # largest p should not be rejected

    def test_benjamini_hochberg(self):
        p_values = np.array([0.001, 0.01, 0.04, 0.5])
        adjusted, reject = benjamini_hochberg(p_values, alpha=0.05)
        assert reject[0]
        assert not reject[3]

    def test_fisher_combine(self):
        p_values = np.array([0.01, 0.02, 0.03])
        stat, combined_p = fisher_combine_pvalues(p_values)
        assert combined_p < 0.05

    def test_stouffer_combine(self):
        p_values = np.array([0.01, 0.02, 0.03])
        z, combined_p = stouffer_combine_pvalues(p_values)
        assert combined_p < 0.05
        assert z > 0

    def test_harmonic_mean_pvalue(self):
        p_values = np.array([0.01, 0.5, 0.8])
        hmp = harmonic_mean_pvalue(p_values)
        assert 0.0 < hmp < 1.0


class TestEffectSize:
    """Tests for effect size calculations."""

    def test_cohens_d_zero(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        d = effect_size_cohens_d(x, y)
        assert abs(d) < 0.2

    def test_cohens_d_large(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(3, 1, 500)
        d = effect_size_cohens_d(x, y)
        assert abs(d) > 2.5

    def test_glass_delta(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(2, 0.5, 500)
        delta = effect_size_glass_delta(x, y)
        assert abs(delta) > 1.5


class TestCollusionPremium:
    """Tests for collusion premium calculation."""

    def test_competitive_premium_near_zero(self):
        rng = np.random.RandomState(42)
        prices = rng.normal(1.05, 0.1, (1000, 2))
        prices = np.clip(prices, 0, None)
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert result["collusion_index"] < 0.15

    def test_collusive_premium_near_one(self):
        prices = np.ones((1000, 2)) * 5.4
        result = run_analysis(prices, nash_price=1.0, monopoly_price=5.5)
        assert result["collusion_index"] > 0.9

    def test_interval_collusion_premium(self):
        obs = Interval(3.0, 3.5)
        nash = Interval(0.9, 1.1)
        mono = Interval(5.3, 5.7)
        premium = interval_collusion_premium(obs, nash, mono)
        # Should be roughly (3.25 - 1.0) / (5.5 - 1.0) ≈ 0.5
        assert premium.contains(0.5) or abs(premium.midpoint - 0.5) < 0.3


class TestTimeSeries:
    """Tests for time series analysis utilities."""

    def test_stationarity_stationary(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 1000)
        dw = durbin_watson(data)
        assert 1.5 < dw < 2.5  # near 2 for no autocorrelation

    def test_stationarity_autocorrelated(self):
        rng = np.random.RandomState(42)
        data = np.zeros(1000)
        data[0] = rng.normal()
        for i in range(1, 1000):
            data[i] = 0.9 * data[i - 1] + rng.normal(0, 0.1)
        dw = durbin_watson(data)
        assert dw < 0.5  # strong positive autocorrelation

    def test_autocorrelation_ljung_box(self):
        rng = np.random.RandomState(42)
        data = np.zeros(500)
        data[0] = rng.normal()
        for i in range(1, 500):
            data[i] = 0.8 * data[i - 1] + rng.normal(0, 0.1)
        Q, p = ljung_box_test(data, max_lag=10)
        assert p < 0.05  # should detect autocorrelation

    def test_moving_average(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = moving_average(data, window=3)
        assert len(ma) == 3
        assert abs(ma[0] - 2.0) < 1e-10
        assert abs(ma[1] - 3.0) < 1e-10
        assert abs(ma[2] - 4.0) < 1e-10

    def test_ema(self):
        data = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
        ema = exponential_moving_average(data, alpha=0.5)
        assert ema[0] == 1.0
        assert ema[-1] > 5.0  # should be approaching 10


class TestPowerAnalysis:
    """Tests for statistical power analysis."""

    def test_sample_size_effect(self):
        """Larger samples should yield more precise bootstrap CIs."""
        rng = np.random.RandomState(42)
        data_small = rng.normal(5.0, 1.0, 50)
        data_large = rng.normal(5.0, 1.0, 5000)

        lo_s, _, hi_s = bootstrap_mean(data_small, n_bootstrap=1000, random_state=42)
        lo_l, _, hi_l = bootstrap_mean(data_large, n_bootstrap=1000, random_state=42)

        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small

    def test_power_increases_with_effect(self):
        """Tests with larger effects should reject more often."""
        rng = np.random.RandomState(42)
        reject_small = 0
        reject_large = 0
        n_reps = 100

        for i in range(n_reps):
            seed = 1000 + i
            rs = np.random.RandomState(seed)
            x = rs.normal(0, 1, 50)
            y_small = rs.normal(0.2, 1, 50)
            y_large = rs.normal(2.0, 1, 50)

            _, p_s = permutation_test(x, y_small, lambda a, b: np.mean(a) - np.mean(b),
                                      n_permutations=200, random_state=seed)
            _, p_l = permutation_test(x, y_large, lambda a, b: np.mean(a) - np.mean(b),
                                      n_permutations=200, random_state=seed)
            if p_s < 0.05:
                reject_small += 1
            if p_l < 0.05:
                reject_large += 1

        assert reject_large > reject_small


class TestRobustStatistics:
    """Tests for robust statistical estimators."""

    def test_trimmed_mean(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        tm = trimmed_mean(data, proportion=0.2)
        # With 10 elements and 20% trimming, removes 2 from each end
        # so trimmed data is [3,4,5,6,7,8], mean=5.5 vs original mean=14.5
        assert tm < np.mean(data)

    def test_winsorized_mean(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        wm = winsorized_mean(data, proportion=0.2)
        # Winsorized: extremes clamped, not removed
        assert wm < np.mean(data)

    def test_robust_std(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 1000)
        rs = robust_std(data)
        assert abs(rs - 1.0) < 0.2

    def test_weighted_mean(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 0.0, 0.0])
        assert abs(weighted_mean(values, weights) - 1.0) < 1e-10

    def test_weighted_variance(self):
        values = np.array([1.0, 2.0, 3.0])
        wv = weighted_variance(values)
        assert wv > 0


class TestNullDistributions:
    """Tests that verify behavior under null hypothesis."""

    def test_p_values_uniform_under_null(self):
        """Under the null, p-values should be approximately uniform."""
        rng = np.random.RandomState(42)
        p_values = []
        for i in range(200):
            x = rng.normal(0, 1, 50)
            y = rng.normal(0, 1, 50)
            _, p = permutation_test(x, y, lambda a, b: np.mean(a) - np.mean(b),
                                    n_permutations=200, random_state=i)
            p_values.append(p)

        p_values = np.array(p_values)
        # Under null, ~5% should be < 0.05
        reject_rate = np.mean(p_values < 0.05)
        assert reject_rate < 0.15  # allow generous margin

    def test_interval_mean_contains_true(self):
        rng = np.random.RandomState(42)
        true_mean = 5.0
        coverage = 0
        n_reps = 100
        for i in range(n_reps):
            data = rng.normal(true_mean, 1.0, 200)
            iv = interval_mean(data)
            if iv.contains(true_mean):
                coverage += 1
        assert coverage / n_reps >= 0.85


class TestKSTest:
    """Tests for the Kolmogorov-Smirnov test."""

    def test_same_distribution(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        d = ks_test_statistic(x, y)
        assert d < 0.1

    def test_different_distribution(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(3, 1, 500)
        d = ks_test_statistic(x, y)
        assert d > 0.5

    def test_empirical_cdf(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sorted_data, cdf_vals = empirical_cdf(data)
        assert sorted_data[0] == 1.0
        assert abs(cdf_vals[-1] - 1.0) < 1e-10
