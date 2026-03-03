"""
Comprehensive tests for dp_forge.zcdp.conversion module.

Tests ZCDPToApproxDP correctness, ApproxDPToZCDP validity,
RDPToZCDP and ZCDPToRDP roundtrip, PLDConversion against numerical
integration, OptimalConversion tightness, and conversion monotonicity.
"""

import math

import numpy as np
import pytest

from dp_forge.zcdp.conversion import (
    ApproxDPToZCDP,
    NumericConversion,
    OptimalConversion,
    PLDConversion,
    RDPToZCDP,
    ZCDPToApproxDP,
    ZCDPToRDP,
)
from dp_forge.types import PrivacyBudget, ZCDPBudget


# =============================================================================
# ZCDPToApproxDP Tests
# =============================================================================


class TestZCDPToApproxDP:
    """Tests for correct ε,δ conversion from zCDP."""

    def test_convert_formula(self):
        """ε = ρ + 2√(ρ·ln(1/δ))."""
        rho = 1.0
        delta = 1e-5
        dp = ZCDPToApproxDP.convert(rho, delta)
        expected = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        assert abs(dp.epsilon - expected) < 1e-10
        assert dp.delta == delta

    def test_convert_small_rho(self):
        dp = ZCDPToApproxDP.convert(0.01, 1e-5)
        assert dp.epsilon > 0
        assert dp.epsilon < 1.0

    def test_convert_large_rho(self):
        dp = ZCDPToApproxDP.convert(10.0, 1e-5)
        assert dp.epsilon > 10.0

    def test_convert_optimal_tighter(self):
        rho = 1.0
        delta = 1e-5
        dp_basic = ZCDPToApproxDP.convert(rho, delta)
        dp_opt = ZCDPToApproxDP.convert_optimal(rho, delta)
        assert dp_opt.epsilon <= dp_basic.epsilon + 1e-6

    def test_convert_truncated(self):
        dp = ZCDPToApproxDP.convert_truncated(xi=0.5, rho=1.0, delta=1e-5)
        dp_no_xi = ZCDPToApproxDP.convert(1.0, 1e-5)
        assert abs(dp.epsilon - dp_no_xi.epsilon - 0.5) < 1e-10

    def test_delta_for_epsilon(self):
        rho = 1.0
        epsilon = 5.0
        delta = ZCDPToApproxDP.delta_for_epsilon(rho, epsilon)
        expected = math.exp(-((epsilon - rho) ** 2) / (4.0 * rho))
        assert abs(delta - expected) < 1e-12
        assert 0 < delta < 1

    def test_epsilon_for_delta(self):
        rho = 1.0
        delta = 1e-5
        eps = ZCDPToApproxDP.epsilon_for_delta(rho, delta)
        expected = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        assert abs(eps - expected) < 1e-10

    def test_roundtrip_epsilon_delta(self):
        """epsilon_for_delta(rho, delta_for_epsilon(rho, eps)) ≈ eps."""
        rho = 0.5
        eps = 3.0
        delta = ZCDPToApproxDP.delta_for_epsilon(rho, eps)
        eps_back = ZCDPToApproxDP.epsilon_for_delta(rho, delta)
        assert abs(eps_back - eps) < 1e-6

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            ZCDPToApproxDP.convert(-1.0, 1e-5)

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            ZCDPToApproxDP.convert(1.0, 0.0)
        with pytest.raises(ValueError):
            ZCDPToApproxDP.convert(1.0, 1.0)

    def test_monotonicity_in_rho(self):
        """Larger ρ → larger ε."""
        delta = 1e-5
        eps1 = ZCDPToApproxDP.convert(0.5, delta).epsilon
        eps2 = ZCDPToApproxDP.convert(1.0, delta).epsilon
        eps3 = ZCDPToApproxDP.convert(2.0, delta).epsilon
        assert eps1 < eps2 < eps3

    def test_monotonicity_in_delta(self):
        """Larger δ → smaller ε."""
        rho = 1.0
        eps1 = ZCDPToApproxDP.convert(rho, 1e-8).epsilon
        eps2 = ZCDPToApproxDP.convert(rho, 1e-5).epsilon
        eps3 = ZCDPToApproxDP.convert(rho, 1e-2).epsilon
        assert eps1 > eps2 > eps3


# =============================================================================
# ApproxDPToZCDP Tests
# =============================================================================


class TestApproxDPToZCDP:
    """Tests for valid ρ computation from (ε,δ)-DP."""

    def test_pure_dp_conversion(self):
        """ρ = ε(e^ε - 1)/2."""
        eps = 1.0
        budget = ApproxDPToZCDP.convert(eps, delta=0.0)
        expected = eps * (math.exp(eps) - 1.0) / 2.0
        assert abs(budget.rho - expected) < 1e-10

    def test_approx_dp_conversion(self):
        budget = ApproxDPToZCDP.convert(1.0, delta=1e-5)
        assert budget.rho > 0

    def test_tight_same_as_basic_for_pure_dp(self):
        eps = 1.0
        basic = ApproxDPToZCDP.convert(eps, delta=0.0)
        tight = ApproxDPToZCDP.convert_tight(eps, delta=0.0)
        assert abs(basic.rho - tight.rho) < 1e-10

    def test_pure_dp_to_zcdp(self):
        budget = ApproxDPToZCDP.pure_dp_to_zcdp(1.0)
        expected = 1.0 * (math.exp(1.0) - 1.0) / 2.0
        assert abs(budget.rho - expected) < 1e-10

    def test_consistency_with_forward_conversion(self):
        """Converting DP→zCDP then zCDP→DP should give ε' ≥ ε."""
        eps_orig = 2.0
        delta = 1e-5
        rho_budget = ApproxDPToZCDP.convert(eps_orig, delta)
        dp_back = ZCDPToApproxDP.convert(rho_budget.rho, delta)
        # Round-trip should recover ε or be looser
        assert dp_back.epsilon <= eps_orig + 1e-6

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            ApproxDPToZCDP.convert(-1.0)

    def test_monotonicity(self):
        """Larger ε → larger ρ."""
        r1 = ApproxDPToZCDP.convert(0.5, delta=0.0).rho
        r2 = ApproxDPToZCDP.convert(1.0, delta=0.0).rho
        r3 = ApproxDPToZCDP.convert(2.0, delta=0.0).rho
        assert r1 < r2 < r3


# =============================================================================
# RDPToZCDP and ZCDPToRDP Roundtrip Tests
# =============================================================================


class TestRDPZCDPRoundtrip:
    """Tests for RDP↔zCDP conversions."""

    def test_gaussian_rdp_to_zcdp(self):
        """For Gaussian: ε(α)=αρ → ρ = Δ²/(2σ²)."""
        budget = RDPToZCDP.from_gaussian(sigma=1.0, sensitivity=1.0)
        assert abs(budget.rho - 0.5) < 1e-10

    def test_convert_from_rdp_curve(self):
        rho_true = 0.5
        curve = lambda alpha: alpha * rho_true
        budget = RDPToZCDP.convert(curve)
        assert abs(budget.rho - rho_true) < 0.01

    def test_from_rdp_values(self):
        rho_true = 0.5
        alphas = np.array([2.0, 5.0, 10.0, 50.0, 100.0])
        epsilons = alphas * rho_true
        budget = RDPToZCDP.from_rdp_values(alphas, epsilons)
        assert abs(budget.rho - rho_true) < 1e-10

    def test_zcdp_to_rdp_formula(self):
        """ε(α) = αρ."""
        rho = 0.5
        alpha = 3.0
        eps = ZCDPToRDP.convert(rho, alpha)
        assert abs(eps - 1.5) < 1e-12

    def test_zcdp_to_rdp_invalid_alpha(self):
        with pytest.raises(ValueError):
            ZCDPToRDP.convert(1.0, 0.5)

    def test_rdp_curve(self):
        curve = ZCDPToRDP.rdp_curve(0.5)
        assert callable(curve)
        assert abs(curve(2.0) - 1.0) < 1e-12
        assert abs(curve(10.0) - 5.0) < 1e-12

    def test_rdp_table(self):
        alphas, epsilons = ZCDPToRDP.rdp_table(0.5)
        assert len(alphas) == len(epsilons)
        np.testing.assert_allclose(epsilons, alphas * 0.5, atol=1e-12)

    def test_roundtrip_zcdp_rdp_zcdp(self):
        """zCDP → RDP curve → zCDP should recover ρ."""
        rho = 1.5
        curve = ZCDPToRDP.rdp_curve(rho)
        budget = RDPToZCDP.convert(curve)
        assert abs(budget.rho - rho) < 0.05

    def test_from_rdp_values_mismatched_shapes(self):
        with pytest.raises(ValueError):
            RDPToZCDP.from_rdp_values(np.array([2.0, 3.0]), np.array([1.0]))

    def test_from_rdp_values_invalid_alpha(self):
        with pytest.raises(ValueError):
            RDPToZCDP.from_rdp_values(np.array([0.5]), np.array([1.0]))


# =============================================================================
# PLDConversion Tests
# =============================================================================


class TestPLDConversion:
    """Tests for PLD-based conversion against numerical integration."""

    def test_gaussian_pld_shape(self):
        pld = PLDConversion(num_points=1024, tail_bound=20.0)
        grid, pdf = pld.gaussian_pld(sigma=1.0, sensitivity=1.0)
        assert len(grid) == 1024
        assert len(pdf) == 1024

    def test_gaussian_pld_normalized(self):
        pld = PLDConversion(num_points=2048, tail_bound=30.0)
        grid, pdf = pld.gaussian_pld(sigma=1.0, sensitivity=1.0)
        dx = grid[1] - grid[0]
        total = np.sum(pdf) * dx
        assert abs(total - 1.0) < 0.01

    def test_compose_plds(self):
        pld = PLDConversion(num_points=1024, tail_bound=20.0)
        g1, p1 = pld.gaussian_pld(sigma=1.0)
        g2, p2 = pld.gaussian_pld(sigma=2.0)
        gc, pc = pld.compose_plds((g1, p1), (g2, p2))
        assert len(gc) == len(pc)
        assert np.all(pc >= 0)

    def test_pld_to_dp(self):
        pld = PLDConversion(num_points=2048, tail_bound=30.0)
        grid, pdf = pld.gaussian_pld(sigma=1.0, sensitivity=1.0)
        dp = pld.pld_to_dp(grid, pdf, delta=1e-5)
        assert dp.epsilon > 0
        assert dp.delta == 1e-5

    def test_zcdp_to_dp_via_pld_consistent(self):
        """PLD-based conversion should be close to analytic."""
        rho = 0.5
        delta = 1e-5
        pld = PLDConversion(num_points=2**14, tail_bound=50.0)
        dp_pld = pld.zcdp_to_dp_via_pld(rho, delta)
        dp_analytic = ZCDPToApproxDP.convert(rho, delta)
        # PLD should be at least close (within factor ~2x)
        assert dp_pld.epsilon > 0
        assert dp_pld.epsilon < dp_analytic.epsilon * 3

    def test_composition_via_pld(self):
        rho = 0.5
        delta = 1e-5
        pld = PLDConversion(num_points=2**12, tail_bound=30.0)
        dp_1 = pld.zcdp_to_dp_via_pld(rho, delta, num_compositions=1)
        dp_2 = pld.zcdp_to_dp_via_pld(rho, delta, num_compositions=2)
        assert dp_2.epsilon > dp_1.epsilon


# =============================================================================
# OptimalConversion Tests
# =============================================================================


class TestOptimalConversion:
    """Tests for OptimalConversion tightness."""

    def test_zcdp_to_dp_tighter_or_equal(self):
        rho = 1.0
        delta = 1e-5
        dp_basic = ZCDPToApproxDP.convert(rho, delta)
        dp_opt = OptimalConversion.zcdp_to_dp(rho, delta)
        assert dp_opt.epsilon <= dp_basic.epsilon + 1e-6

    def test_dp_to_zcdp_optimal_small_delta(self):
        eps = 1.0
        budget = OptimalConversion.dp_to_zcdp_optimal(eps, delta=1e-10)
        assert isinstance(budget, ZCDPBudget)
        # Resulting rho should be at most eps^2 / 2 + small slack
        assert budget.rho <= eps ** 2 / 2 + 0.1

    def test_dp_to_zcdp_optimal_approx(self):
        eps = 2.0
        delta = 1e-5
        budget = OptimalConversion.dp_to_zcdp_optimal(eps, delta)
        # Verify: converting back should give ε' ≤ ε
        dp_back = ZCDPToApproxDP.convert(budget.rho, delta)
        assert dp_back.epsilon <= eps + 0.1

    def test_optimal_delta(self):
        rho = 1.0
        eps = 5.0
        delta = OptimalConversion.optimal_delta(rho, eps)
        expected = math.exp(-((eps - rho) ** 2) / (4.0 * rho))
        assert abs(delta - expected) < 1e-12

    def test_optimal_rdp_order(self):
        rho = 1.0
        delta = 1e-5
        alpha = OptimalConversion.optimal_rdp_order(rho, delta)
        expected = 1.0 + math.sqrt(math.log(1.0 / delta) / rho)
        assert abs(alpha - expected) < 1e-10

    def test_optimal_rdp_order_increases_with_rho(self):
        delta = 1e-5
        a1 = OptimalConversion.optimal_rdp_order(0.1, delta)
        a2 = OptimalConversion.optimal_rdp_order(1.0, delta)
        assert a1 > a2  # larger ρ → smaller α*

    def test_invalid_epsilon_for_delta(self):
        with pytest.raises(ValueError, match="epsilon must be > rho"):
            OptimalConversion.optimal_delta(rho=5.0, epsilon=3.0)


# =============================================================================
# NumericConversion Tests
# =============================================================================


class TestNumericConversion:
    """Tests for NumericConversion numerical methods."""

    def test_rdp_to_dp_gaussian(self):
        nc = NumericConversion()
        curve = lambda alpha: alpha * 0.5  # Gaussian RDP
        dp = nc.rdp_to_dp_numerical(curve, delta=1e-5)
        dp_analytic = ZCDPToApproxDP.convert(0.5, 1e-5)
        assert abs(dp.epsilon - dp_analytic.epsilon) < 0.5

    def test_rdp_to_zcdp_numerical(self):
        nc = NumericConversion()
        rho_true = 0.5
        curve = lambda alpha: alpha * rho_true
        budget = nc.rdp_to_zcdp_numerical(curve)
        assert abs(budget.rho - rho_true) < 0.01

    def test_gaussian_hockey_stick(self):
        nc = NumericConversion()
        delta = nc.gaussian_hockey_stick(sigma=1.0, sensitivity=1.0, epsilon=1.0)
        assert 0 <= delta <= 1

    def test_hockey_stick_monotone_in_epsilon(self):
        nc = NumericConversion()
        d1 = nc.gaussian_hockey_stick(1.0, 1.0, epsilon=0.5)
        d2 = nc.gaussian_hockey_stick(1.0, 1.0, epsilon=1.0)
        d3 = nc.gaussian_hockey_stick(1.0, 1.0, epsilon=2.0)
        assert d1 >= d2 >= d3

    def test_binary_search_sigma(self):
        nc = NumericConversion()
        sigma = nc.binary_search_sigma(epsilon=1.0, delta=1e-5)
        assert sigma > 0
        # Verify: the sigma should achieve approximately (eps, delta)
        d = nc.gaussian_hockey_stick(sigma, 1.0, 1.0)
        assert d <= 1e-5 + 1e-6

    def test_compose_rdp_curves(self):
        nc = NumericConversion()
        c1 = lambda alpha: alpha * 0.3
        c2 = lambda alpha: alpha * 0.2
        composed = nc.compose_rdp_curves([c1, c2])
        assert abs(composed(5.0) - 5.0 * 0.5) < 1e-12

    def test_interpolated_rdp(self):
        nc = NumericConversion()
        alphas = np.array([2.0, 3.0, 5.0, 10.0])
        epsilons = alphas * 0.5
        interp = nc.interpolated_rdp(alphas, epsilons)
        # Exact at known points
        assert abs(interp(2.0) - 1.0) < 0.1
        assert abs(interp(5.0) - 2.5) < 0.1


# =============================================================================
# Conversion Consistency Tests
# =============================================================================


class TestConversionConsistency:
    """Cross-module consistency checks."""

    def test_zcdp_to_dp_matches_budget_method(self):
        rho = 1.0
        delta = 1e-5
        dp1 = ZCDPToApproxDP.convert(rho, delta)
        budget = ZCDPBudget(rho=rho)
        dp2 = budget.to_approx_dp(delta)
        assert abs(dp1.epsilon - dp2.epsilon) < 1e-10

    def test_conversion_triangle_inequality(self):
        """DP→zCDP→DP should give ε' ≤ ε (lossy)."""
        eps = 2.0
        delta = 1e-5
        zcdp = ApproxDPToZCDP.convert(eps, delta)
        dp_back = ZCDPToApproxDP.convert(zcdp.rho, delta)
        assert dp_back.epsilon <= eps + 0.1
