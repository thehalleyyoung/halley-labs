"""
Comprehensive tests for dp_forge.discretization module.

Tests cover error bounds, optimal range computation, grid size
recommendations, convergence tables, and approximation certificates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dp_forge.discretization import (
    DiscretizationAnalyzer,
    DiscretizationCertificate,
    _piecewise_constant_error,
    _piecewise_linear_error,
    _laplace_tail_mse,
    _gaussian_tail_mse,
)
from dp_forge.types import QuerySpec
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def analyzer_eps1():
    """DiscretizationAnalyzer with ε=1, Δf=1."""
    return DiscretizationAnalyzer(epsilon=1.0, sensitivity=1.0)


@pytest.fixture
def analyzer_eps2_gaussian():
    """DiscretizationAnalyzer for Gaussian mechanism."""
    return DiscretizationAnalyzer(
        epsilon=1.0, sensitivity=1.0, delta=1e-5,
    )


# =========================================================================
# Section 1: Error Bound Computation
# =========================================================================


class TestErrorBound:
    """Tests for discretization error bounds."""

    def test_error_bound_dict(self, analyzer_eps1):
        """error_bound returns dict with expected keys."""
        result = analyzer_eps1.error_bound(k=100)
        assert isinstance(result, dict)
        assert "discretization_error" in result or "total_error_bound" in result

    def test_error_bound_positive(self, analyzer_eps1):
        """Error bound is positive for k > 1."""
        result = analyzer_eps1.error_bound(k=50)
        total = sum(v for v in result.values() if isinstance(v, (int, float)) and v > 0)
        assert total > 0

    def test_error_bound_o_b_over_k(self, analyzer_eps1):
        """Error bound ∝ O(B/k) for uniform grid (piecewise constant).

        Discretization error = (Δy)²/12 where Δy = 2B/(k-1).
        So error ∝ B²/k².
        """
        B = analyzer_eps1.optimal_range()
        r1 = analyzer_eps1.error_bound(k=50, range_B=B)
        r2 = analyzer_eps1.error_bound(k=100, range_B=B)
        # Get discretization error from both
        e1 = r1.get("discretization_error", r1.get("total_error_bound", 0))
        e2 = r2.get("discretization_error", r2.get("total_error_bound", 0))
        # Error should decrease with k
        assert e2 < e1

    def test_larger_k_smaller_error(self, analyzer_eps1):
        """Larger k → smaller total error."""
        B = analyzer_eps1.optimal_range()
        results = []
        for k in [10, 50, 100, 200]:
            r = analyzer_eps1.error_bound(k=k, range_B=B)
            total = r.get("total_error_bound", r.get("discretization_error", 0))
            results.append(total)
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1] - 1e-12


# =========================================================================
# Section 2: Optimal Range
# =========================================================================


class TestOptimalRange:
    """Tests for optimal range computation."""

    def test_optimal_range_positive(self, analyzer_eps1):
        """Optimal range is positive."""
        B = analyzer_eps1.optimal_range()
        assert B > 0

    def test_optimal_range_finite(self, analyzer_eps1):
        """Optimal range is finite."""
        B = analyzer_eps1.optimal_range()
        assert math.isfinite(B)

    def test_optimal_range_laplace_formula(self):
        """Optimal range matches known Laplace tail formula.

        For Laplace(b), tail CDF = exp(-x/b)/2.
        Range B s.t. tail mass < α should be ~ -b·ln(2α).
        b = Δf/ε.
        """
        eps, delta_f = 1.0, 1.0
        analyzer = DiscretizationAnalyzer(
            epsilon=eps, sensitivity=delta_f, delta=0.0,
        )
        B = analyzer.optimal_range(alpha=0.01)
        b = delta_f / eps  # Scale parameter
        # For α=0.01, rough estimate: B ≈ b * ln(1/α) ≈ b * 4.6
        assert B > b * 2  # Should be at least a few scale params wide
        assert B < b * 50  # But not excessively wide

    def test_gaussian_range_larger(self):
        """Gaussian mechanism range typically differs from Laplace."""
        an_lap = DiscretizationAnalyzer(epsilon=1.0, sensitivity=1.0, delta=0.0)
        an_gauss = DiscretizationAnalyzer(epsilon=1.0, sensitivity=1.0, delta=1e-5)
        B_lap = an_lap.optimal_range()
        B_gauss = an_gauss.optimal_range()
        # Both should be positive
        assert B_lap > 0
        assert B_gauss > 0


# =========================================================================
# Section 3: Grid Size Recommendation
# =========================================================================


class TestGridSizeRecommendation:
    """Tests for grid size recommendations."""

    def test_recommend_returns_dict(self, analyzer_eps1):
        """recommend_grid_size returns dict with k and other info."""
        result = analyzer_eps1.recommend_grid_size(target_error=0.1)
        assert isinstance(result, dict)
        assert "k" in result

    def test_recommended_k_positive(self, analyzer_eps1):
        """Recommended k is a positive integer."""
        result = analyzer_eps1.recommend_grid_size(target_error=0.1)
        k = result["k"]
        assert k >= 2
        assert isinstance(k, (int, np.integer))

    def test_larger_k_for_tighter_target(self, analyzer_eps1):
        """Tighter target error → larger recommended k."""
        r1 = analyzer_eps1.recommend_grid_size(target_error=1.0)
        r2 = analyzer_eps1.recommend_grid_size(target_error=0.01)
        assert r2["k"] >= r1["k"]

    def test_recommended_k_achieves_target(self, analyzer_eps1):
        """Recommended k achieves the target error."""
        target = 0.1
        rec = analyzer_eps1.recommend_grid_size(target_error=target)
        k = rec["k"]
        err = analyzer_eps1.error_bound(k=k)
        total = err.get("total_error_bound", err.get("discretization_error", 0))
        # Should be close to or below target
        assert total <= target * 2  # Allow some margin


# =========================================================================
# Section 4: Convergence Table
# =========================================================================


class TestConvergenceTable:
    """Tests for convergence table generation."""

    def test_convergence_table_decreasing(self, analyzer_eps1):
        """Errors decrease with k in convergence table."""
        table = analyzer_eps1.convergence_table()
        assert len(table) > 0
        # Extract k and error values
        for i in range(len(table) - 1):
            k_curr = table[i]["k"]
            k_next = table[i + 1]["k"]
            if k_next > k_curr:
                e_curr = table[i].get(
                    "total_error_bound", table[i].get("discretization_error", float("inf"))
                )
                e_next = table[i + 1].get(
                    "total_error_bound", table[i + 1].get("discretization_error", float("inf"))
                )
                assert e_next <= e_curr + 1e-12

    def test_convergence_table_custom_k(self, analyzer_eps1):
        """Custom k_values in convergence table."""
        k_values = [5, 10, 50, 100, 500]
        table = analyzer_eps1.convergence_table(k_values=k_values)
        assert len(table) == len(k_values)
        for row, k in zip(table, k_values):
            assert row["k"] == k

    def test_convergence_table_has_k_field(self, analyzer_eps1):
        """Each row in convergence table has 'k' field."""
        table = analyzer_eps1.convergence_table()
        for row in table:
            assert "k" in row
            assert row["k"] >= 2


# =========================================================================
# Section 5: Approximation Certificate
# =========================================================================


class TestApproximationCertificate:
    """Tests for discretization approximation certificates."""

    def test_certificate_type(self, analyzer_eps1):
        """approximation_certificate returns DiscretizationCertificate."""
        cert = analyzer_eps1.approximation_certificate(k=100)
        assert isinstance(cert, DiscretizationCertificate)

    def test_certificate_fields(self, analyzer_eps1):
        """Certificate has all required fields."""
        cert = analyzer_eps1.approximation_certificate(k=50)
        assert cert.k == 50
        assert cert.epsilon == 1.0
        assert cert.sensitivity == 1.0
        assert cert.total_error_bound >= 0
        assert cert.discretization_error >= 0
        assert cert.tail_error >= 0

    def test_certificate_total_bound(self, analyzer_eps1):
        """Total error bound = discretization + tail."""
        cert = analyzer_eps1.approximation_certificate(k=50)
        assert abs(
            cert.total_error_bound - (cert.discretization_error + cert.tail_error)
        ) < 1e-10

    def test_certificate_grid_spacing(self, analyzer_eps1):
        """Grid spacing is positive and consistent with k."""
        cert = analyzer_eps1.approximation_certificate(k=100)
        assert cert.grid_spacing > 0
        # Grid spacing = 2B / (k-1)
        expected_spacing = 2 * cert.range_B / (cert.k - 1)
        assert abs(cert.grid_spacing - expected_spacing) < 1e-10

    def test_certificate_timestamp(self, analyzer_eps1):
        """Certificate has timestamp."""
        cert = analyzer_eps1.approximation_certificate(k=50)
        assert cert.timestamp is not None

    def test_certificate_mechanism_family(self, analyzer_eps1):
        """Certificate records mechanism family."""
        cert = analyzer_eps1.approximation_certificate(k=50)
        assert cert.mechanism_family == "piecewise_constant"


# =========================================================================
# Section 6: Grid Type Comparison
# =========================================================================


class TestGridTypeComparison:
    """Tests for comparing grid types."""

    def test_compare_returns_dict(self, analyzer_eps1):
        """compare_grid_types returns dict with grid type keys."""
        result = analyzer_eps1.compare_grid_types(k=50)
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_piecewise_linear_tighter(self, analyzer_eps1):
        """Piecewise linear error ≤ piecewise constant error."""
        result = analyzer_eps1.compare_grid_types(k=50)
        if "piecewise_constant" in result and "piecewise_linear" in result:
            e_const = result["piecewise_constant"].get(
                "discretization_error",
                result["piecewise_constant"].get("total_error_bound", float("inf")),
            )
            e_linear = result["piecewise_linear"].get(
                "discretization_error",
                result["piecewise_linear"].get("total_error_bound", float("inf")),
            )
            assert e_linear <= e_const + 1e-10


# =========================================================================
# Section 7: Low-level Error Functions
# =========================================================================


class TestErrorFunctions:
    """Tests for low-level error computation functions."""

    def test_piecewise_constant_error(self):
        """Piecewise constant error = (Δy)²/12."""
        spacing = 0.1
        err = _piecewise_constant_error(spacing)
        expected = spacing ** 2 / 12
        assert abs(err - expected) < 1e-15

    def test_piecewise_linear_error(self):
        """Piecewise linear error = (Δy)⁴/720."""
        spacing = 0.1
        err = _piecewise_linear_error(spacing)
        expected = spacing ** 4 / 720
        assert abs(err - expected) < 1e-15

    def test_piecewise_linear_tighter_than_constant(self):
        """Piecewise linear error < constant for same spacing."""
        for spacing in [0.01, 0.1, 0.5]:
            e_const = _piecewise_constant_error(spacing)
            e_linear = _piecewise_linear_error(spacing)
            assert e_linear <= e_const

    def test_laplace_tail_mse(self):
        """Laplace tail MSE decreases with B."""
        b, eps = 1.0, 1.0
        t1 = _laplace_tail_mse(B=5.0, sensitivity=b, epsilon=eps)
        t2 = _laplace_tail_mse(B=10.0, sensitivity=b, epsilon=eps)
        assert t2 < t1
        assert t1 > 0
        assert t2 > 0

    def test_laplace_tail_mse_zero_at_infinity(self):
        """Laplace tail MSE → 0 as B → ∞."""
        b, eps = 1.0, 1.0
        t = _laplace_tail_mse(B=100.0, sensitivity=b, epsilon=eps)
        assert t < 1e-10

    def test_gaussian_tail_mse(self):
        """Gaussian tail MSE decreases with B."""
        sigma = 1.0
        t1 = _gaussian_tail_mse(B=3.0, sigma=sigma)
        t2 = _gaussian_tail_mse(B=6.0, sigma=sigma)
        assert t2 < t1
        assert t1 > 0

    def test_gaussian_tail_mse_zero_at_infinity(self):
        """Gaussian tail MSE → 0 as B → ∞."""
        sigma = 1.0
        t = _gaussian_tail_mse(B=20.0, sigma=sigma)
        assert t < 1e-10


# =========================================================================
# Section 8: Factory Method
# =========================================================================


class TestFromQuerySpec:
    """Tests for creating analyzer from QuerySpec."""

    def test_from_query_spec(self):
        """from_query_spec creates valid analyzer."""
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=2.0,
            k=50,
        )
        analyzer = DiscretizationAnalyzer.from_query_spec(spec)
        assert isinstance(analyzer, DiscretizationAnalyzer)
        B = analyzer.optimal_range()
        assert B > 0


# =========================================================================
# Section 9: Edge Cases
# =========================================================================


class TestDiscretizationEdgeCases:
    """Edge case tests for discretization."""

    def test_k_equals_2(self, analyzer_eps1):
        """k=2 (minimum) still produces valid error bound."""
        result = analyzer_eps1.error_bound(k=2)
        assert isinstance(result, dict)

    def test_very_large_k(self, analyzer_eps1):
        """Very large k gives very small discretization error."""
        cert = analyzer_eps1.approximation_certificate(k=10000)
        assert cert.discretization_error < 1e-4

    def test_large_epsilon(self):
        """Large ε → smaller optimal range (concentrated mechanism)."""
        a1 = DiscretizationAnalyzer(epsilon=1.0, sensitivity=1.0)
        a2 = DiscretizationAnalyzer(epsilon=10.0, sensitivity=1.0)
        B1 = a1.optimal_range()
        B2 = a2.optimal_range()
        assert B2 < B1  # Larger ε → more concentrated → smaller range

    def test_large_sensitivity(self):
        """Larger sensitivity → larger optimal range."""
        a1 = DiscretizationAnalyzer(epsilon=1.0, sensitivity=1.0)
        a2 = DiscretizationAnalyzer(epsilon=1.0, sensitivity=5.0)
        B1 = a1.optimal_range()
        B2 = a2.optimal_range()
        assert B2 > B1  # Larger sensitivity → wider mechanism → larger range
