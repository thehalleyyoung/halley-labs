"""Tests for usability_oracle.montecarlo.variance_reduction.

Verifies control variates, antithetic variates, stratified sampling,
effective sample size, and Rao–Blackwell improvement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.montecarlo.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
    StratifiedSampling,
    compute_effective_sample_size,
    rao_blackwellize,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# =====================================================================
# Control variates
# =====================================================================

class TestControlVariates:
    """Test variance reduction via control variates."""

    def test_reduces_variance(self, rng: np.random.Generator) -> None:
        """Control variates should reduce estimator variance when correlated.

        We use X = f(U) and Z = U (uniform), E[Z] = 0.5.
        """
        n = 1000
        u = rng.uniform(0, 1, n)
        x = np.exp(u)  # E[e^U] ≈ e - 1
        z = u           # control variate: E[U] = 0.5

        cv = ControlVariates()
        adjusted, c_star, ratio = cv.apply(x.tolist(), z.tolist(), 0.5)

        var_original = np.var(x, ddof=1)
        var_adjusted = np.var(adjusted, ddof=1)
        assert var_adjusted < var_original

    def test_uncorrelated_no_reduction(self, rng: np.random.Generator) -> None:
        """If X and Z are independent, variance reduction ratio ≈ 1."""
        n = 500
        x = rng.normal(0, 1, n)
        z = rng.normal(0, 1, n)  # independent
        cv = ControlVariates()
        _, _, ratio = cv.apply(x.tolist(), z.tolist(), 0.0)
        # Ratio should be close to 1 (no reduction)
        assert ratio > 0.8

    def test_single_sample_no_crash(self) -> None:
        """Single sample should return copy without error."""
        cv = ControlVariates()
        adjusted, c, ratio = cv.apply([5.0], [3.0], 2.0)
        assert len(adjusted) == 1
        assert ratio == 1.0

    def test_optimal_coefficient_sign(self, rng: np.random.Generator) -> None:
        """Optimal c should be positive for positively correlated X, Z."""
        n = 200
        u = rng.uniform(0, 1, n)
        x = 2.0 * u + rng.normal(0, 0.1, n)
        z = u
        cv = ControlVariates()
        _, c_star, _ = cv.apply(x.tolist(), z.tolist(), 0.5)
        assert c_star > 0


# =====================================================================
# Antithetic variates
# =====================================================================

class TestAntitheticVariates:
    """Test variance reduction via antithetic variates."""

    def test_produces_negatively_correlated_pairs(
        self, rng: np.random.Generator
    ) -> None:
        """Paired means should have lower variance than originals.

        We test using f(U) and f(1-U) where f is monotone, which guarantees
        negative correlation between originals and antithetics.
        """
        n = 500
        u = rng.uniform(0, 1, n)
        originals = np.exp(u)
        antithetics = np.exp(1 - u)

        av = AntitheticVariates()
        paired, ratio = av.pair_estimates(originals.tolist(), antithetics.tolist())

        # Ratio should be < 1 (variance reduced)
        assert ratio < 1.0
        assert len(paired) == n

    def test_empty_returns_empty(self) -> None:
        """Empty input should return empty output."""
        av = AntitheticVariates()
        paired, ratio = av.pair_estimates([], [])
        assert len(paired) == 0
        assert ratio == 1.0

    def test_paired_mean_correct(self) -> None:
        """Paired mean should be (X + X') / 2."""
        av = AntitheticVariates()
        originals = [2.0, 4.0, 6.0]
        antithetics = [3.0, 5.0, 1.0]
        paired, _ = av.pair_estimates(originals, antithetics)
        expected = np.array([2.5, 4.5, 3.5])
        np.testing.assert_allclose(paired, expected)


# =====================================================================
# Stratified sampling
# =====================================================================

class TestStratifiedSampling:
    """Test stratified sampling variance reduction."""

    def test_covers_all_strata(self) -> None:
        """Allocation should assign samples to every stratum."""
        ss = StratifiedSampling()
        strata = {"low": 0.3, "mid": 0.4, "high": 0.3}
        allocation = ss.allocate_samples(strata, total_samples=30)
        assert set(allocation.keys()) == {"low", "mid", "high"}
        for k, v in allocation.items():
            assert v >= 1

    def test_total_samples_correct(self) -> None:
        """Total allocated samples should equal the budget."""
        ss = StratifiedSampling()
        strata = {"a": 0.5, "b": 0.3, "c": 0.2}
        allocation = ss.allocate_samples(strata, total_samples=100)
        assert sum(allocation.values()) == 100

    def test_proportional_allocation(self) -> None:
        """Strata with higher weight should get more samples."""
        ss = StratifiedSampling()
        strata = {"heavy": 0.8, "light": 0.2}
        allocation = ss.allocate_samples(strata, total_samples=100)
        assert allocation["heavy"] > allocation["light"]

    def test_empty_strata(self) -> None:
        """Empty strata dict should return empty allocation."""
        ss = StratifiedSampling()
        assert ss.allocate_samples({}, total_samples=10) == {}

    def test_combine_strata(self) -> None:
        """Combining strata should give a weighted average."""
        ss = StratifiedSampling()
        means = {"a": 2.0, "b": 4.0}
        variances = {"a": 1.0, "b": 1.0}
        weights = {"a": 0.5, "b": 0.5}
        sizes = {"a": 50, "b": 50}
        overall_mean, overall_var = ss.combine_strata(means, variances, weights, sizes)
        assert overall_mean == pytest.approx(3.0)
        assert overall_var > 0


# =====================================================================
# Effective sample size
# =====================================================================

class TestEffectiveSampleSize:
    """Test ESS computation from importance weights."""

    def test_ess_le_n(self) -> None:
        """ESS should always be ≤ n."""
        weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        ess = compute_effective_sample_size(weights)
        assert ess <= len(weights) + 1e-10

    def test_ess_equals_n_for_uniform(self) -> None:
        """ESS = n when all weights are equal."""
        n = 10
        weights = np.ones(n) / n
        ess = compute_effective_sample_size(weights)
        assert ess == pytest.approx(n, abs=1e-10)

    def test_ess_equals_1_for_degenerate(self) -> None:
        """ESS = 1 when only one weight is non-zero."""
        weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        ess = compute_effective_sample_size(weights)
        assert ess == pytest.approx(1.0, abs=1e-10)

    def test_ess_zero_for_empty(self) -> None:
        """ESS = 0 for empty weight array."""
        ess = compute_effective_sample_size(np.array([]))
        assert ess == 0.0

    @pytest.mark.parametrize("n", [5, 20, 100])
    def test_ess_between_1_and_n(self, n: int) -> None:
        """ESS is in [1, n] for any valid normalised weights."""
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n))
        ess = compute_effective_sample_size(weights)
        assert 1.0 - 1e-10 <= ess <= n + 1e-10


# =====================================================================
# Rao-Blackwell
# =====================================================================

class TestRaoBlackwell:
    """Test Rao–Blackwell improvement."""

    def test_reduces_variance(self, rng: np.random.Generator) -> None:
        """Rao-Blackwellized estimates should have ≤ variance."""
        n = 200
        x = rng.normal(5.0, 2.0, n)
        stat = x + rng.normal(0, 0.1, n)  # correlated statistic
        improved = rao_blackwellize(x.tolist(), {"stat": stat.tolist()})
        # Mean should be preserved approximately
        assert np.mean(improved) == pytest.approx(np.mean(x), abs=0.5)
        # Variance should not increase (in expectation)
        assert np.var(improved) <= np.var(x) + 0.5  # allow tolerance

    def test_single_sample_returns_copy(self) -> None:
        """Single sample should return a copy."""
        result = rao_blackwellize([3.14], {"s": [1.0]})
        assert len(result) == 1
        assert result[0] == pytest.approx(3.14)

    def test_empty_statistics_returns_copy(self) -> None:
        """No sufficient statistics → return copy."""
        x = [1.0, 2.0, 3.0]
        result = rao_blackwellize(x, {})
        np.testing.assert_array_equal(result, x)
