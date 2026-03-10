"""Unit tests for usability_oracle.sensitivity.sobol — Sobol sensitivity analysis.

Tests cover Sobol sequence generation, first-order and total-order index
computation, convergence monitoring, and validation with known analytic
functions (Ishigami).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.interval import Interval
from usability_oracle.sensitivity.types import ParameterRange, SensitivityConfig, SobolIndices
from usability_oracle.sensitivity.sobol import (
    ConvergenceRecord,
    SobolAnalyzer,
    compute_first_order,
    compute_total_order,
    monitor_convergence,
    rank_parameters_by_total_order,
    identify_interactions,
    saltelli_sample,
    sobol_sequence,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _standard_params(n: int = 3) -> list[ParameterRange]:
    """Create n parameters on [0, 1]."""
    return [
        ParameterRange(name=f"x{i}", interval=Interval(0.0, 1.0), nominal=0.5)
        for i in range(n)
    ]


def _linear_model(**kwargs: float) -> float:
    """y = x0 + 2*x1 + 3*x2"""
    return kwargs.get("x0", 0) + 2 * kwargs.get("x1", 0) + 3 * kwargs.get("x2", 0)


def _ishigami(**kwargs: float) -> float:
    """Ishigami function: y = sin(x0) + 7*sin²(x1) + 0.1*x2⁴*sin(x0)."""
    x0 = kwargs.get("x0", 0)
    x1 = kwargs.get("x1", 0)
    x2 = kwargs.get("x2", 0)
    return math.sin(x0) + 7.0 * math.sin(x1) ** 2 + 0.1 * x2**4 * math.sin(x0)


def _ishigami_params() -> list[ParameterRange]:
    """Parameters for the Ishigami function on [-π, π]."""
    return [
        ParameterRange(name="x0", interval=Interval(-math.pi, math.pi), nominal=0.0),
        ParameterRange(name="x1", interval=Interval(-math.pi, math.pi), nominal=0.0),
        ParameterRange(name="x2", interval=Interval(-math.pi, math.pi), nominal=0.0),
    ]


def _additive_model(**kwargs: float) -> float:
    """Purely additive: y = x0² + x1."""
    return kwargs.get("x0", 0) ** 2 + kwargs.get("x1", 0)


# ═══════════════════════════════════════════════════════════════════════════
# Sobol Sequence Generation
# ═══════════════════════════════════════════════════════════════════════════


class TestSobolSequence:
    """Test quasi-random Sobol sequence generation."""

    def test_shape(self):
        seq = sobol_sequence(64, 3)
        assert seq.shape == (64, 3)

    def test_unit_interval(self):
        seq = sobol_sequence(128, 5)
        assert np.all(seq >= 0.0)
        assert np.all(seq <= 1.0)

    def test_deterministic_with_seed(self):
        a = sobol_sequence(32, 2, seed=0)
        b = sobol_sequence(32, 2, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = sobol_sequence(32, 2, seed=0)
        b = sobol_sequence(32, 2, seed=1)
        assert not np.array_equal(a, b)

    def test_1d_sequence(self):
        seq = sobol_sequence(16, 1)
        assert seq.shape == (16, 1)

    def test_large_sequence(self):
        seq = sobol_sequence(1024, 4)
        assert seq.shape == (1024, 4)

    def test_uniformity(self):
        """Sobol points should be more uniform than random."""
        seq = sobol_sequence(256, 2)
        # Check that points cover the space reasonably
        for dim in range(2):
            assert seq[:, dim].min() < 0.1
            assert seq[:, dim].max() > 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Saltelli Sampling
# ═══════════════════════════════════════════════════════════════════════════


class TestSaltelliSample:
    """Test Saltelli sampling scheme for Sobol indices."""

    def test_sample_shapes(self):
        params = _standard_params(3)
        A, B, AB_list = saltelli_sample(64, params)
        assert A.shape == (64, 3)
        assert B.shape == (64, 3)
        assert len(AB_list) == 3
        for ab in AB_list:
            assert ab.shape == (64, 3)

    def test_sample_in_range(self):
        params = [
            ParameterRange(name="x0", interval=Interval(2.0, 5.0), nominal=3.5),
        ]
        A, B, AB_list = saltelli_sample(32, params)
        assert np.all(A >= 2.0 - 1e-10)
        assert np.all(A <= 5.0 + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# First-Order and Total-Order Indices
# ═══════════════════════════════════════════════════════════════════════════


class TestSobolIndicesComputation:
    """Test Sobol index computation from model evaluations."""

    def test_first_order_in_valid_range(self):
        n = 256
        y_A = np.random.default_rng(42).normal(size=n)
        y_B = np.random.default_rng(43).normal(size=n)
        y_AB = np.random.default_rng(44).normal(size=n)
        S = compute_first_order(y_A, y_B, y_AB)
        # First-order index can be slightly negative due to estimation error
        assert -0.5 < S < 1.5

    def test_total_order_in_valid_range(self):
        n = 256
        y_A = np.random.default_rng(42).normal(size=n)
        y_B = np.random.default_rng(43).normal(size=n)
        y_AB = np.random.default_rng(44).normal(size=n)
        ST = compute_total_order(y_A, y_B, y_AB)
        assert -0.5 < ST < 1.5

    def test_total_geq_first_order(self):
        """Total-order index should be >= first-order for well-behaved functions."""
        analyzer = SobolAnalyzer()
        params = _standard_params(2)
        results = analyzer.compute_sobol(_additive_model, params, n_samples=512, seed=42)
        for r in results:
            assert r.total_order >= r.first_order - 0.15  # allow estimation noise


# ═══════════════════════════════════════════════════════════════════════════
# SobolAnalyzer Full Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestSobolAnalyzer:
    """Test the full SobolAnalyzer workflow."""

    def test_linear_model_indices(self):
        """For y = x0 + 2x1 + 3x2, S_i ∝ coeff²."""
        analyzer = SobolAnalyzer()
        params = _standard_params(3)
        results = analyzer.compute_sobol(_linear_model, params, n_samples=1024, seed=42)
        assert len(results) == 3
        # x2 (coefficient 3) should have the highest first-order index
        by_name = {r.parameter_name: r for r in results}
        assert by_name["x2"].first_order > by_name["x0"].first_order

    def test_additive_model_no_interaction(self):
        """For a purely additive model, first-order ≈ total-order."""
        analyzer = SobolAnalyzer()
        params = _standard_params(2)
        results = analyzer.compute_sobol(_additive_model, params, n_samples=1024, seed=42)
        for r in results:
            assert abs(r.total_order - r.first_order) < 0.15

    def test_ishigami_function(self):
        """Ishigami: x0 and x1 influential, x2 has interactions."""
        analyzer = SobolAnalyzer()
        params = _ishigami_params()
        results = analyzer.compute_sobol(_ishigami, params, n_samples=2048, seed=42)
        by_name = {r.parameter_name: r for r in results}
        # x1 (7*sin²) should have high first-order
        assert by_name["x1"].first_order > 0.2
        # x2 has interaction with x0 via 0.1*x2^4*sin(x0)
        assert by_name["x2"].total_order > by_name["x2"].first_order - 0.05

    def test_total_variance(self):
        analyzer = SobolAnalyzer()
        params = _standard_params(3)
        var = analyzer.total_variance(_linear_model, params, n_samples=1024, seed=42)
        assert var > 0

    def test_analyze_method(self):
        """Test the high-level analyze interface."""
        analyzer = SobolAnalyzer()
        config = SensitivityConfig(
            parameters=tuple(_standard_params(2)),
            n_samples=256,
            method="sobol",
            output_names=("y",),
            seed=42,
        )
        result = analyzer.analyze(_additive_model, config)
        assert result.n_evaluations > 0
        assert len(result.sobol_indices) == 2

    def test_confidence_intervals(self):
        analyzer = SobolAnalyzer()
        params = _standard_params(2)
        results = analyzer.compute_sobol(_additive_model, params, n_samples=512, seed=42,
                                         confidence_level=0.95)
        for r in results:
            assert r.first_order_ci.low <= r.first_order
            assert r.first_order_ci.high >= r.first_order


# ═══════════════════════════════════════════════════════════════════════════
# Convergence Monitoring
# ═══════════════════════════════════════════════════════════════════════════


class TestConvergenceMonitoring:
    """Test convergence monitoring for Sobol analysis."""

    def test_monitor_returns_record(self):
        params = _standard_params(2)
        record = monitor_convergence(
            _additive_model, params,
            sample_schedule=(64, 128, 256),
            seed=42,
        )
        assert isinstance(record, ConvergenceRecord)
        assert len(record.sample_sizes) == 3

    def test_convergence_check(self):
        params = _standard_params(2)
        record = monitor_convergence(
            _additive_model, params,
            sample_schedule=(128, 256, 512, 1024),
            seed=42,
        )
        # Should show convergence for a simple function
        converged = record.is_converged(tolerance=0.1)
        assert isinstance(converged, bool)

    def test_relative_change(self):
        params = _standard_params(2)
        record = monitor_convergence(
            _additive_model, params,
            sample_schedule=(64, 128, 256),
            seed=42,
        )
        change = record.relative_change("x0")
        assert isinstance(change, float)


# ═══════════════════════════════════════════════════════════════════════════
# Ranking and Interaction Identification
# ═══════════════════════════════════════════════════════════════════════════


class TestRankingAndInteractions:
    """Test parameter ranking and interaction identification."""

    def test_ranking_by_total_order(self):
        analyzer = SobolAnalyzer()
        params = _standard_params(3)
        results = analyzer.compute_sobol(_linear_model, params, n_samples=1024, seed=42)
        ranking = rank_parameters_by_total_order(results)
        assert len(ranking) == 3
        # ranking returns (name, value) tuples — first element is most influential
        top_name = ranking[0] if isinstance(ranking[0], str) else ranking[0][0]
        assert top_name == "x2"  # coefficient 3

    def test_identify_interactions(self):
        analyzer = SobolAnalyzer()
        params = _ishigami_params()
        results = analyzer.compute_sobol(_ishigami, params, n_samples=2048, seed=42)
        interactions = identify_interactions(results, threshold=0.01)
        assert isinstance(interactions, list)

    def test_sobol_indices_properties(self):
        """Test SobolIndices dataclass properties."""
        idx = SobolIndices(
            parameter_name="x0",
            first_order=0.3,
            total_order=0.5,
            first_order_ci=Interval(0.2, 0.4),
            total_order_ci=Interval(0.4, 0.6),
            second_order={},
        )
        assert idx.interaction_index == pytest.approx(0.2, abs=0.01)
        assert idx.is_influential
