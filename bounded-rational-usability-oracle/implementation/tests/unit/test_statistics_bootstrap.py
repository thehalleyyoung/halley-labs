"""Unit tests for usability_oracle.statistics.bootstrap — Bootstrap CI methods.

Tests percentile CI, BCa CI, CI narrowing, and block bootstrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.statistics.bootstrap import BootstrapCI
from usability_oracle.statistics.types import BootstrapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mean_stat(data: np.ndarray) -> float:
    return float(np.mean(data))


def _median_stat(data: np.ndarray) -> float:
    return float(np.median(data))


# ===================================================================
# Percentile CI
# ===================================================================


class TestPercentileCI:

    def test_contains_true_mean(self):
        """For normal data with known mean, the 95% CI should contain the true mean."""
        rng = _rng()
        data = rng.normal(5.0, 1.0, 200)
        bs = BootstrapCI(n_bootstrap=2000, seed=42)
        result = bs.percentile_ci(data, _mean_stat, alpha=0.05)
        assert isinstance(result, BootstrapResult)
        assert result.ci.lower <= 5.0 <= result.ci.upper

    def test_ci_bounds_ordered(self):
        rng = _rng()
        data = rng.normal(0, 1, 100)
        bs = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bs.percentile_ci(data, _mean_stat)
        assert result.ci.lower <= result.ci.upper

    def test_point_estimate_inside_ci(self):
        rng = _rng()
        data = rng.normal(0, 1, 100)
        bs = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bs.percentile_ci(data, _mean_stat)
        assert result.ci.lower <= result.ci.point_estimate <= result.ci.upper

    def test_with_median_statistic(self):
        rng = _rng()
        data = rng.normal(3.0, 2.0, 100)
        bs = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bs.percentile_ci(data, _median_stat)
        assert result.ci.lower <= result.ci.upper


# ===================================================================
# BCa CI
# ===================================================================


class TestBCaCI:

    def test_produces_valid_ci(self):
        rng = _rng()
        data = rng.normal(0, 1, 100)
        bs = BootstrapCI(n_bootstrap=1000, seed=42)
        result = bs.bca_ci(data, _mean_stat)
        assert isinstance(result, BootstrapResult)
        assert result.ci.lower <= result.ci.upper

    def test_contains_true_mean(self):
        rng = _rng()
        data = rng.normal(10.0, 2.0, 200)
        bs = BootstrapCI(n_bootstrap=2000, seed=42)
        result = bs.bca_ci(data, _mean_stat, alpha=0.05)
        assert result.ci.lower <= 10.0 <= result.ci.upper

    def test_bca_vs_percentile_widths(self):
        """On average, BCa should not be drastically wider than percentile."""
        rng = _rng(7)
        data = rng.normal(0, 1, 150)
        bs = BootstrapCI(n_bootstrap=2000, seed=7)
        pct = bs.percentile_ci(data, _mean_stat)
        bca = bs.bca_ci(data, _mean_stat)
        pct_width = pct.ci.upper - pct.ci.lower
        bca_width = bca.ci.upper - bca.ci.lower
        # BCa width should be in a reasonable ratio
        assert bca_width < pct_width * 3


# ===================================================================
# CI narrows with more data
# ===================================================================


class TestCINarrowing:

    def test_ci_narrows_with_more_data(self):
        rng = _rng()
        bs = BootstrapCI(n_bootstrap=1000, seed=42)

        data_small = rng.normal(0, 1, 30)
        data_large = rng.normal(0, 1, 300)

        ci_small = bs.percentile_ci(data_small, _mean_stat)
        ci_large = bs.percentile_ci(data_large, _mean_stat)

        width_small = ci_small.ci.upper - ci_small.ci.lower
        width_large = ci_large.ci.upper - ci_large.ci.lower
        assert width_large < width_small


# ===================================================================
# Block bootstrap (time-series aware)
# ===================================================================


class TestBlockBootstrap:

    def test_handles_time_series(self):
        rng = _rng()
        # AR(1)-like time series
        n = 100
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = 0.5 * data[i - 1] + rng.normal(0, 1)
        bs = BootstrapCI(n_bootstrap=500, seed=42)
        result = bs.block_bootstrap(data, _mean_stat, block_size=5)
        assert isinstance(result, BootstrapResult)
        assert result.ci.lower <= result.ci.upper

    def test_block_size_1_like_regular(self):
        rng = _rng()
        data = rng.normal(0, 1, 50)
        bs = BootstrapCI(n_bootstrap=500, seed=42)
        result = bs.block_bootstrap(data, _mean_stat, block_size=1)
        assert result.ci.lower <= result.ci.upper

    def test_ci_bounds_finite(self):
        rng = _rng()
        data = rng.normal(0, 1, 60)
        bs = BootstrapCI(n_bootstrap=500, seed=42)
        result = bs.block_bootstrap(data, _mean_stat, block_size=10)
        assert np.isfinite(result.ci.lower)
        assert np.isfinite(result.ci.upper)


# ===================================================================
# Edge cases
# ===================================================================


class TestBootstrapEdgeCases:

    def test_small_sample(self):
        data = np.array([1.0, 2.0, 3.0])
        bs = BootstrapCI(n_bootstrap=500, seed=42)
        result = bs.percentile_ci(data, _mean_stat)
        assert result.ci.lower <= result.ci.upper

    def test_constant_data(self):
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        bs = BootstrapCI(n_bootstrap=500, seed=42)
        result = bs.percentile_ci(data, _mean_stat)
        np.testing.assert_allclose(result.ci.lower, 5.0, atol=1e-10)
        np.testing.assert_allclose(result.ci.upper, 5.0, atol=1e-10)
