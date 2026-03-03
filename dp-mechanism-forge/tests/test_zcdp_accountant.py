"""
Comprehensive tests for dp_forge.zcdp.accountant module.

Tests RenyiDivergenceComputer correctness, GaussianMechanismZCDP,
LaplaceMechanismZCDP bounds, SubsampledZCDP amplification,
and AdvancedCompositionZCDP additivity.
"""

import math

import numpy as np
import pytest

from dp_forge.zcdp.accountant import (
    AdvancedCompositionZCDP,
    GaussianMechanismZCDP,
    LaplaceMechanismZCDP,
    RenyiDivergenceComputer,
    SubsampledZCDP,
)
from dp_forge.types import PrivacyBudget, ZCDPBudget


# =============================================================================
# RenyiDivergenceComputer Tests
# =============================================================================


class TestRenyiDivergenceComputer:
    """Tests for Rényi divergence computations."""

    def test_gaussian_known_value(self):
        """D_α(N(0,1) || N(1,1)) = α/2."""
        d = RenyiDivergenceComputer.gaussian(alpha=2.0, sigma=1.0, sensitivity=1.0)
        assert abs(d - 1.0) < 1e-12  # α·Δ²/(2σ²) = 2·1/(2·1) = 1

    def test_gaussian_scaling_with_sigma(self):
        d1 = RenyiDivergenceComputer.gaussian(2.0, sigma=1.0)
        d2 = RenyiDivergenceComputer.gaussian(2.0, sigma=2.0)
        assert abs(d2 / d1 - 0.25) < 1e-12  # σ² scales inversely

    def test_gaussian_scaling_with_alpha(self):
        d2 = RenyiDivergenceComputer.gaussian(2.0, sigma=1.0)
        d5 = RenyiDivergenceComputer.gaussian(5.0, sigma=1.0)
        assert abs(d5 / d2 - 2.5) < 1e-12

    def test_gaussian_zero_sensitivity(self):
        d = RenyiDivergenceComputer.gaussian(2.0, sigma=1.0, sensitivity=0.0)
        assert d == 0.0

    def test_gaussian_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            RenyiDivergenceComputer.gaussian(0.5, sigma=1.0)

    def test_gaussian_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            RenyiDivergenceComputer.gaussian(2.0, sigma=-1.0)

    def test_laplace_known_value(self):
        """Check Laplace RDP for α=2, b=1, Δ=1."""
        d = RenyiDivergenceComputer.laplace(alpha=2.0, scale=1.0, sensitivity=1.0)
        # D_2 = 1/(2-1) * log(2/(2·2-1)·e^{(2-1)·1} + (2-1)/(2·2-1)·e^{-2·1})
        t1 = math.log(2/3) + 1.0
        t2 = math.log(1/3) - 2.0
        expected = (math.log(math.exp(t1) + math.exp(t2)))
        assert abs(d - expected) < 1e-10

    def test_laplace_zero_sensitivity(self):
        d = RenyiDivergenceComputer.laplace(2.0, scale=1.0, sensitivity=0.0)
        assert d == 0.0

    def test_laplace_positive(self):
        d = RenyiDivergenceComputer.laplace(3.0, scale=2.0, sensitivity=1.0)
        assert d > 0

    def test_discrete_gaussian_close_to_continuous(self):
        """For large σ², discrete ≈ continuous."""
        d_cont = RenyiDivergenceComputer.gaussian(2.0, sigma=10.0, sensitivity=1)
        d_disc = RenyiDivergenceComputer.discrete_gaussian(2.0, sigma_sq=100.0, sensitivity=1)
        assert abs(d_cont - d_disc) < 1e-6

    def test_from_distributions_identical(self):
        """D_α(P || P) = 0."""
        p = np.array([0.3, 0.5, 0.2])
        d = RenyiDivergenceComputer.from_distributions(2.0, p, p)
        assert abs(d) < 1e-10

    def test_from_distributions_diverging(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        d = RenyiDivergenceComputer.from_distributions(2.0, p, q)
        assert d > 0

    def test_from_distributions_infinite_support_mismatch(self):
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.0, 0.5, 0.5])
        d = RenyiDivergenceComputer.from_distributions(2.0, p, q)
        # p>0 where q=0 → infinite divergence
        assert d == float("inf")

    def test_rdp_curve_gaussian(self):
        comp = RenyiDivergenceComputer()
        curve = comp.rdp_curve("gaussian", {"sigma": 1.0, "sensitivity": 1.0})
        assert callable(curve)
        val = curve(3.0)
        expected = 3.0 * 1.0 / 2.0
        assert abs(val - expected) < 1e-12

    def test_rdp_curve_laplace(self):
        comp = RenyiDivergenceComputer()
        curve = comp.rdp_curve("laplace", {"scale": 1.0})
        val = curve(2.0)
        assert val > 0

    def test_rdp_curve_unknown_raises(self):
        comp = RenyiDivergenceComputer()
        with pytest.raises(ValueError, match="Unknown"):
            comp.rdp_curve("unknown", {})


# =============================================================================
# GaussianMechanismZCDP Tests
# =============================================================================


class TestGaussianMechanismZCDP:
    """Tests for Gaussian mechanism under zCDP."""

    def test_zcdp_cost_formula(self):
        """ρ = Δ²/(2σ²)."""
        gm = GaussianMechanismZCDP(sigma=2.0, sensitivity=1.0)
        expected = 1.0 / (2.0 * 4.0)
        assert abs(gm.zcdp_cost() - expected) < 1e-12

    def test_zcdp_cost_with_sensitivity(self):
        gm = GaussianMechanismZCDP(sigma=1.0, sensitivity=2.0)
        expected = 4.0 / 2.0
        assert abs(gm.zcdp_cost() - expected) < 1e-12

    def test_zcdp_cost_override_sensitivity(self):
        gm = GaussianMechanismZCDP(sigma=1.0, sensitivity=1.0)
        rho = gm.zcdp_cost(sensitivity=3.0)
        assert abs(rho - 9.0 / 2.0) < 1e-12

    def test_rdp(self):
        gm = GaussianMechanismZCDP(sigma=1.0, sensitivity=1.0)
        # ε(α) = α·Δ²/(2σ²) = α/2
        assert abs(gm.rdp(3.0) - 1.5) < 1e-12

    def test_rdp_invalid_alpha(self):
        gm = GaussianMechanismZCDP(sigma=1.0)
        with pytest.raises(ValueError):
            gm.rdp(0.5)

    def test_from_rho(self):
        gm = GaussianMechanismZCDP.from_rho(rho=0.5, sensitivity=1.0)
        assert abs(gm.zcdp_cost() - 0.5) < 1e-12

    def test_from_rho_sigma(self):
        gm = GaussianMechanismZCDP.from_rho(rho=0.5, sensitivity=1.0)
        expected_sigma = 1.0 / math.sqrt(1.0)
        assert abs(gm.sigma - expected_sigma) < 1e-12

    def test_optimal_sigma(self):
        sigma = GaussianMechanismZCDP.optimal_sigma(0.5, sensitivity=1.0)
        assert abs(sigma - 1.0) < 1e-12

    def test_to_approx_dp(self):
        gm = GaussianMechanismZCDP(sigma=1.0, sensitivity=1.0)
        dp = gm.to_approx_dp(delta=1e-5)
        assert isinstance(dp, PrivacyBudget)
        assert dp.epsilon > 0
        assert dp.delta == 1e-5

    def test_name(self):
        gm = GaussianMechanismZCDP(sigma=1.0)
        assert gm.name == "gaussian"

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            GaussianMechanismZCDP(sigma=0.0)

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            GaussianMechanismZCDP.from_rho(rho=-1.0)


# =============================================================================
# LaplaceMechanismZCDP Tests
# =============================================================================


class TestLaplaceMechanismZCDP:
    """Tests for Laplace mechanism zCDP bounds."""

    def test_pure_dp_epsilon(self):
        lm = LaplaceMechanismZCDP(scale=2.0, sensitivity=1.0)
        assert abs(lm.pure_dp_epsilon - 0.5) < 1e-12

    def test_zcdp_cost_positive(self):
        lm = LaplaceMechanismZCDP(scale=1.0, sensitivity=1.0)
        rho = lm.zcdp_cost()
        assert rho > 0

    def test_zcdp_cost_bounded_by_eps_sq(self):
        """ρ ≤ ε²/2 for Laplace."""
        lm = LaplaceMechanismZCDP(scale=2.0, sensitivity=1.0)
        eps = lm.pure_dp_epsilon
        rho = lm.zcdp_cost()
        assert rho <= eps**2 / 2.0 + 1e-10

    def test_rdp_positive(self):
        lm = LaplaceMechanismZCDP(scale=1.0, sensitivity=1.0)
        d = lm.rdp(2.0)
        assert d > 0

    def test_from_rho(self):
        lm = LaplaceMechanismZCDP.from_rho(rho=0.5)
        assert lm.scale > 0
        assert lm.sensitivity == 1.0

    def test_to_approx_dp(self):
        lm = LaplaceMechanismZCDP(scale=1.0, sensitivity=1.0)
        dp = lm.to_approx_dp(delta=1e-5)
        assert dp.epsilon > 0

    def test_name(self):
        lm = LaplaceMechanismZCDP(scale=1.0)
        assert lm.name == "laplace"

    def test_invalid_scale(self):
        with pytest.raises(ValueError):
            LaplaceMechanismZCDP(scale=0.0)

    def test_more_noise_less_rho(self):
        lm1 = LaplaceMechanismZCDP(scale=1.0, sensitivity=1.0)
        lm2 = LaplaceMechanismZCDP(scale=2.0, sensitivity=1.0)
        assert lm2.zcdp_cost() < lm1.zcdp_cost()


# =============================================================================
# SubsampledZCDP Tests
# =============================================================================


class TestSubsampledZCDP:
    """Tests for SubsampledZCDP amplification."""

    def test_full_sampling_no_amplification(self):
        sub = SubsampledZCDP(base_rho=1.0, sampling_rate=1.0)
        assert abs(sub.amplified_rho() - 1.0) < 1e-6

    def test_subsampling_reduces_rho(self):
        sub = SubsampledZCDP(base_rho=1.0, sampling_rate=0.1)
        assert sub.amplified_rho() < 1.0

    def test_smaller_rate_less_rho(self):
        sub1 = SubsampledZCDP(base_rho=1.0, sampling_rate=0.5)
        sub2 = SubsampledZCDP(base_rho=1.0, sampling_rate=0.1)
        assert sub2.amplified_rho() < sub1.amplified_rho()

    def test_amplified_rho_positive(self):
        sub = SubsampledZCDP(base_rho=0.5, sampling_rate=0.01)
        assert sub.amplified_rho() > 0

    def test_to_budget(self):
        sub = SubsampledZCDP(base_rho=1.0, sampling_rate=0.5)
        budget = sub.to_budget()
        assert isinstance(budget, ZCDPBudget)
        assert budget.rho > 0

    def test_invalid_base_rho(self):
        with pytest.raises(ValueError):
            SubsampledZCDP(base_rho=-1.0, sampling_rate=0.5)

    def test_invalid_sampling_rate(self):
        with pytest.raises(ValueError):
            SubsampledZCDP(base_rho=1.0, sampling_rate=0.0)
        with pytest.raises(ValueError):
            SubsampledZCDP(base_rho=1.0, sampling_rate=1.5)


# =============================================================================
# AdvancedCompositionZCDP Tests
# =============================================================================


class TestAdvancedCompositionZCDP:
    """Tests for AdvancedCompositionZCDP additivity."""

    def test_empty_composition(self):
        comp = AdvancedCompositionZCDP()
        assert comp.total_rho == 0.0
        assert comp.num_mechanisms == 0

    def test_additive_composition(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=0.5)
        comp.add(rho=0.3)
        assert abs(comp.total_rho - 0.8) < 1e-12

    def test_add_gaussian(self):
        comp = AdvancedCompositionZCDP()
        comp.add_gaussian(sigma=1.0, sensitivity=1.0)
        expected = 1.0 / 2.0
        assert abs(comp.total_rho - expected) < 1e-12

    def test_add_laplace(self):
        comp = AdvancedCompositionZCDP()
        comp.add_laplace(scale=1.0, sensitivity=1.0)
        assert comp.total_rho > 0

    def test_add_subsampled(self):
        comp = AdvancedCompositionZCDP()
        comp.add_subsampled(base_rho=1.0, sampling_rate=0.1)
        assert comp.total_rho > 0
        assert comp.total_rho < 1.0

    def test_total_xi(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=1.0, xi=0.1)
        comp.add(rho=0.5, xi=0.2)
        assert abs(comp.total_xi - 0.3) < 1e-12

    def test_get_budget(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=0.5)
        comp.add(rho=0.3)
        budget = comp.get_budget()
        assert isinstance(budget, ZCDPBudget)
        assert abs(budget.rho - 0.8) < 1e-12

    def test_to_approx_dp(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=0.5)
        dp = comp.to_approx_dp(delta=1e-5)
        assert isinstance(dp, PrivacyBudget)
        assert dp.epsilon > 0

    def test_optimal_dp_conversion(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=0.5)
        dp = comp.optimal_dp_conversion(delta=1e-5)
        assert dp.epsilon > 0

    def test_optimal_tighter_than_basic(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=1.0)
        dp_basic = comp.to_approx_dp(delta=1e-5)
        dp_opt = comp.optimal_dp_conversion(delta=1e-5)
        # Optimal should be at least as tight
        assert dp_opt.epsilon <= dp_basic.epsilon + 1e-6

    def test_reset(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=1.0)
        comp.reset()
        assert comp.num_mechanisms == 0
        assert comp.total_rho == 0.0

    def test_summary(self):
        comp = AdvancedCompositionZCDP()
        comp.add(rho=0.5, name="g1")
        comp.add(rho=0.3, name="g2")
        s = comp.summary()
        assert s["num_mechanisms"] == 2
        assert abs(s["total_rho"] - 0.8) < 1e-12
        assert len(s["mechanisms"]) == 2

    def test_invalid_rho(self):
        comp = AdvancedCompositionZCDP()
        with pytest.raises(ValueError):
            comp.add(rho=-1.0)

    def test_composition_many_mechanisms(self):
        comp = AdvancedCompositionZCDP()
        for _ in range(100):
            comp.add_gaussian(sigma=10.0, sensitivity=1.0)
        expected = 100 * (1.0 / 200.0)
        assert abs(comp.total_rho - expected) < 1e-10
