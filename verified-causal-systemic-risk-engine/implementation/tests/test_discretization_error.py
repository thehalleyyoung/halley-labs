"""Tests for discretization error analysis and SMT verification."""

import pytest
import numpy as np


class TestDiscretizationErrorAnalyzer:
    """Test formal discretization error bounds."""

    def _make_analyzer(self):
        from causalbound.junction.discretization_error import (
            DiscretizationErrorAnalyzer,
        )
        return DiscretizationErrorAnalyzer(default_density_bound=5.0)

    def test_uniform_error_bound(self):
        """Uniform discretization TV bound = M * h / 2."""
        analyzer = self._make_analyzer()
        result = analyzer.compute_error_bound(
            variable_name="exposure",
            domain=(0.0, 1.0),
            n_bins=10,
            strategy="uniform",
            density_bound=5.0,
        )
        expected_tv = 5.0 * 0.1 / 2  # M * h / 2 = 0.25
        assert abs(result.tv_distance_bound - expected_tv) < 1e-10
        assert result.n_bins == 10
        assert result.convergence_rate == "O(1/n)"

    def test_quantile_error_bound(self):
        """Quantile discretization TV bound = 1/(2n)."""
        analyzer = self._make_analyzer()
        result = analyzer.compute_error_bound(
            variable_name="loss",
            domain=(0.0, 100.0),
            n_bins=20,
            strategy="quantile",
        )
        expected_tv = 1.0 / (2 * 20)
        assert abs(result.tv_distance_bound - expected_tv) < 1e-10

    def test_error_decreases_with_bins(self):
        """Error should decrease as number of bins increases."""
        analyzer = self._make_analyzer()
        prev_tv = float("inf")
        for n_bins in [4, 8, 16, 32, 64]:
            result = analyzer.compute_error_bound(
                "x", (0.0, 1.0), n_bins, "uniform", 5.0
            )
            assert result.tv_distance_bound < prev_tv
            prev_tv = result.tv_distance_bound

    def test_global_error_subadditivity(self):
        """Global TV <= sum of per-variable TVs."""
        analyzer = self._make_analyzer()
        bounds = [
            analyzer.compute_error_bound(f"v{i}", (0, 1), 10, "uniform", 5.0)
            for i in range(5)
        ]
        global_err = analyzer.compute_global_error(
            bounds, lipschitz_constant=2.0, n_separators=3, tolerance=1.0
        )
        expected_total = sum(b.tv_distance_bound for b in bounds)
        assert abs(global_err.total_tv_bound - expected_total) < 1e-10

    def test_gap_contribution(self):
        """Gap contribution = n_separators * L * total_tv."""
        analyzer = self._make_analyzer()
        bounds = [
            analyzer.compute_error_bound("v0", (0, 1), 20, "uniform", 3.0)
        ]
        global_err = analyzer.compute_global_error(
            bounds, lipschitz_constant=4.0, n_separators=2, tolerance=10.0
        )
        expected_gap = 2 * 4.0 * bounds[0].tv_distance_bound
        assert abs(global_err.composition_gap_contribution - expected_gap) < 1e-10

    def test_adaptive_refinement(self):
        """Adaptive refinement finds minimum bins for target TV."""
        analyzer = self._make_analyzer()
        result = analyzer.adaptive_refinement(
            "exposure", (0.0, 1.0), target_tv=0.01,
            strategy="uniform", density_bound=5.0,
        )
        assert result.tv_distance_bound <= 0.01 + 1e-10
        assert result.n_bins >= 1

    def test_convergence_experiment(self):
        """Convergence experiment should show valid bounds."""
        analyzer = self._make_analyzer()
        result = analyzer.convergence_experiment(
            domain=(0.0, 1.0),
            bin_counts=[4, 8, 16, 32],
            n_samples=10000,
            seed=42,
        )
        assert len(result["bin_counts"]) == 4
        assert len(result["empirical_tv"]) == 4
        assert len(result["bound_tv"]) == 4


class TestDiscretizationVerifier:
    """Test SMT verification of discretization error bounds."""

    def _make_verifier(self):
        from causalbound.smt.discretization_verifier import DiscretizationVerifier
        return DiscretizationVerifier(timeout_ms=5000)

    def test_full_verification(self):
        """Full discretization verification should pass."""
        verifier = self._make_verifier()
        result = verifier.verify_discretization_bounds(
            density_bounds=[5.0, 3.0],
            bin_widths=[0.1, 0.05],
            domain_widths=[1.0, 1.0],
            lipschitz_constant=2.0,
            n_separators=2,
            composed_lower=0.3,
            composed_upper=0.7,
        )
        assert result.all_verified
        assert result.corrected_bounds_verified
        assert result.gap_contribution_verified

    def test_corrected_bounds_wider(self):
        """Corrected bounds should be wider than composed bounds."""
        verifier = self._make_verifier()
        result = verifier.verify_discretization_bounds(
            density_bounds=[4.0],
            bin_widths=[0.1],
            domain_widths=[1.0],
            lipschitz_constant=3.0,
            n_separators=1,
            composed_lower=0.4,
            composed_upper=0.6,
        )
        assert result.corrected_lower <= 0.4
        assert result.corrected_upper >= 0.6
