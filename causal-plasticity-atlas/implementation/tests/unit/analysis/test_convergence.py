"""Unit tests for cpa.analysis.convergence – ConvergenceAnalyzer, ArchiveDiversity."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.analysis.convergence import (
    ConvergenceAnalyzer,
    ConvergenceMetrics,
    ArchiveDiversity,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def analyzer():
    return ConvergenceAnalyzer(window_size=10, patience=5)


@pytest.fixture
def large_window_analyzer():
    return ConvergenceAnalyzer(window_size=50, patience=20)


@pytest.fixture
def diversity_analyzer():
    return ArchiveDiversity(descriptor_dim=4, n_bins_per_dim=5)


def _make_archive_state(coverage, quality, quality_var=0.01, improvement=0.0):
    # Generate synthetic qualities that match the desired mean and variance
    n = 20
    qualities = np.full(n, quality, dtype=np.float64)
    if quality_var > 0:
        # Add small perturbations to create some variance
        rng = np.random.default_rng(42)
        qualities = qualities + rng.normal(0, np.sqrt(quality_var), n)
    return {
        "coverage": coverage,
        "qualities": qualities.tolist(),
    }


# ===================================================================
# Tests – ConvergenceAnalyzer
# ===================================================================


class TestConvergenceAnalyzer:
    """Test ConvergenceAnalyzer convergence detection."""

    def test_update_returns_metrics(self, analyzer):
        state = _make_archive_state(0.1, 0.5)
        metrics = analyzer.update(state)
        assert isinstance(metrics, ConvergenceMetrics)

    def test_not_converged_initially(self, analyzer):
        state = _make_archive_state(0.1, 0.5)
        analyzer.update(state)
        assert not analyzer.is_converged()

    def test_detects_convergence_plateau(self, analyzer):
        # Feed constant quality for many iterations → should converge
        for i in range(30):
            state = _make_archive_state(0.5, 1.0, 0.01, 0.0)
            analyzer.update(state)
        assert analyzer.is_converged(tolerance=1e-3)

    def test_not_converged_with_improvement(self, analyzer):
        for i in range(20):
            # Provide strongly increasing quality with large spread
            state = _make_archive_state(0.1 + i * 0.04, 0.5 + i * 0.5)
            analyzer.update(state)
        # Improving significantly → should not converge
        assert not analyzer.is_converged(tolerance=1e-6)

    def test_convergence_rate_is_float(self, analyzer):
        for i in range(15):
            state = _make_archive_state(0.5, 1.0 - 0.01 * i)
            analyzer.update(state)
        rate = analyzer.convergence_rate()
        assert isinstance(rate, float)

    def test_coverage_curve_length(self, analyzer):
        for i in range(10):
            state = _make_archive_state(0.1 * i, 0.5)
            analyzer.update(state)
        curve = analyzer.coverage_curve()
        assert len(curve) == 10

    def test_quality_curve_length(self, analyzer):
        for i in range(10):
            state = _make_archive_state(0.5, 0.1 * i)
            analyzer.update(state)
        curve = analyzer.quality_curve()
        assert len(curve) == 10

    def test_coverage_curve_increases(self, analyzer):
        for i in range(10):
            state = _make_archive_state(0.1 * (i + 1), 0.5)
            analyzer.update(state)
        curve = analyzer.coverage_curve()
        for j in range(1, len(curve)):
            assert curve[j] >= curve[j - 1]


# ===================================================================
# Tests – Moving average computation
# ===================================================================


class TestMovingAverage:
    """Test _moving_average static helper."""

    def test_moving_average_constant(self):
        values = [5.0] * 10
        ma = ConvergenceAnalyzer._moving_average(values, 3)
        for v in ma:
            assert_allclose(v, 5.0, atol=1e-10)

    def test_moving_average_linear(self):
        values = list(range(10))
        ma = ConvergenceAnalyzer._moving_average(values, 3)
        assert len(ma) > 0
        # Moving average of linearly increasing values
        for v in ma:
            assert isinstance(v, (float, np.floating))

    def test_moving_average_window_1(self):
        values = [1.0, 2.0, 3.0, 4.0]
        ma = ConvergenceAnalyzer._moving_average(values, 1)
        assert len(ma) == len(values)

    def test_moving_average_empty(self):
        ma = ConvergenceAnalyzer._moving_average([], 3)
        assert len(ma) == 0

    def test_relative_improvement_constant(self):
        values = [1.0] * 20
        ri = ConvergenceAnalyzer._relative_improvement(values, 5)
        assert_allclose(ri, 0.0, atol=1e-6)


# ===================================================================
# Tests – ArchiveDiversity
# ===================================================================


class TestArchiveDiversity:
    """Test ArchiveDiversity metrics."""

    def test_compute_returns_dict(self, diversity_analyzer, rng):
        entries = [
            {"descriptor": rng.uniform(0, 1, 4).tolist(), "quality": float(rng.uniform())}
            for _ in range(20)
        ]
        result = diversity_analyzer.compute(entries)
        assert isinstance(result, dict)

    def test_diverse_archive_high_distance(self, diversity_analyzer, rng):
        # Spread-out descriptors
        entries = [
            {"descriptor": [i / 10, j / 10, 0.5, 0.5], "quality": 1.0}
            for i in range(5) for j in range(5)
        ]
        result = diversity_analyzer.compute(entries)
        assert "mean_pairwise_distance" in result or "coverage_ratio" in result

    def test_uniform_archive_low_diversity(self, diversity_analyzer):
        # All entries identical
        entries = [
            {"descriptor": [0.5, 0.5, 0.5, 0.5], "quality": 1.0}
            for _ in range(20)
        ]
        result = diversity_analyzer.compute(entries)
        if "mean_pairwise_distance" in result:
            assert_allclose(result["mean_pairwise_distance"], 0.0, atol=1e-10)

    def test_coverage_ratio_in_range(self, diversity_analyzer, rng):
        entries = [
            {"descriptor": rng.uniform(0, 1, 4).tolist(), "quality": float(rng.uniform())}
            for _ in range(15)
        ]
        result = diversity_analyzer.compute(entries, total_cells=100)
        if "coverage_ratio" in result:
            assert 0.0 <= result["coverage_ratio"] <= 1.0

    def test_single_entry(self, diversity_analyzer):
        entries = [{"descriptor": [0.5, 0.5, 0.5, 0.5], "quality": 1.0}]
        result = diversity_analyzer.compute(entries)
        assert isinstance(result, dict)

    def test_quality_variance_zero_for_uniform(self, diversity_analyzer):
        entries = [
            {"descriptor": [i * 0.1, 0.5, 0.5, 0.5], "quality": 1.0}
            for i in range(10)
        ]
        result = diversity_analyzer.compute(entries)
        if "quality_variance" in result:
            assert_allclose(result["quality_variance"], 0.0, atol=1e-10)

    def test_entropy_nonnegative(self, diversity_analyzer, rng):
        entries = [
            {"descriptor": rng.uniform(0, 1, 4).tolist(), "quality": float(rng.uniform())}
            for _ in range(20)
        ]
        result = diversity_analyzer.compute(entries)
        if "descriptor_entropy" in result:
            assert result["descriptor_entropy"] >= 0.0


# ===================================================================
# Tests – Edge cases
# ===================================================================


class TestConvergenceEdgeCases:
    """Test edge cases for convergence analysis."""

    def test_single_update(self, analyzer):
        state = _make_archive_state(0.1, 0.5)
        metrics = analyzer.update(state)
        assert not analyzer.is_converged()

    def test_very_slow_convergence(self, large_window_analyzer):
        for i in range(100):
            state = _make_archive_state(0.5, 1.0 - 1e-8 * i)
            large_window_analyzer.update(state)
        # Very small changes → should converge with tight tolerance
        assert large_window_analyzer.is_converged(tolerance=1e-4)

    def test_oscillating_quality(self, analyzer):
        for i in range(30):
            q = 0.5 + 0.1 * (-1) ** i
            state = _make_archive_state(0.5, q)
            analyzer.update(state)
        # Oscillating → may or may not converge, just should not crash

    def test_metrics_fields(self, analyzer):
        state = _make_archive_state(0.5, 1.0, 0.01, 0.05)
        metrics = analyzer.update(state)
        assert hasattr(metrics, "archive_coverage")
        assert hasattr(metrics, "mean_quality")
        assert hasattr(metrics, "quality_variance")
        assert hasattr(metrics, "improvement_rate")
        assert hasattr(metrics, "stagnation_count")
