"""
Comprehensive tests for dp_forge.zcdp.synthesis module.

Tests ZCDPSynthesizer mechanism production, GaussianOptimizer optimality,
DiscreteGaussianSynthesis zCDP compliance, MultiQuerySynthesis budget
splitting, and BudgetAllocation optimality.
"""

import math

import numpy as np
import pytest

from dp_forge.zcdp.synthesis import (
    BudgetAllocation,
    DiscreteGaussianSynthesis,
    GaussianOptimizer,
    MultiQuerySynthesis,
    TruncatedGaussianSynthesis,
    ZCDPSynthesizer,
)
from dp_forge.types import PrivacyBudget, ZCDPBudget


# =============================================================================
# GaussianOptimizer Tests
# =============================================================================


class TestGaussianOptimizer:
    """Tests for GaussianOptimizer optimality."""

    def test_sigma_for_rho(self):
        """σ = Δ/√(2ρ)."""
        sigma = GaussianOptimizer.sigma_for_rho(0.5, sensitivity=1.0)
        assert abs(sigma - 1.0) < 1e-12

    def test_rho_for_sigma(self):
        """ρ = Δ²/(2σ²)."""
        rho = GaussianOptimizer.rho_for_sigma(1.0, sensitivity=1.0)
        assert abs(rho - 0.5) < 1e-12

    def test_roundtrip_sigma_rho(self):
        sigma = 2.5
        rho = GaussianOptimizer.rho_for_sigma(sigma, sensitivity=1.0)
        sigma_back = GaussianOptimizer.sigma_for_rho(rho, sensitivity=1.0)
        assert abs(sigma_back - sigma) < 1e-12

    def test_sigma_for_dp(self):
        sigma = GaussianOptimizer.sigma_for_dp(epsilon=1.0, delta=1e-5)
        assert sigma > 0
        # Check it achieves the right rho
        rho = GaussianOptimizer.rho_for_sigma(sigma)
        from dp_forge.zcdp.conversion import ZCDPToApproxDP
        dp = ZCDPToApproxDP.convert(rho, 1e-5)
        assert dp.epsilon <= 1.0 + 0.1

    def test_mse(self):
        assert abs(GaussianOptimizer.mse(2.0) - 4.0) < 1e-12

    def test_mae(self):
        sigma = 2.0
        expected = sigma * math.sqrt(2.0 / math.pi)
        assert abs(GaussianOptimizer.mae(sigma) - expected) < 1e-12

    def test_confidence_interval(self):
        ci = GaussianOptimizer.confidence_interval(1.0, confidence=0.95)
        assert ci > 1.5  # z_0.975 ≈ 1.96

    def test_confidence_interval_invalid(self):
        with pytest.raises(ValueError):
            GaussianOptimizer.confidence_interval(1.0, confidence=0.0)

    def test_optimize_for_utility(self):
        sigma, util = GaussianOptimizer.optimize_for_utility(
            rho=0.5, sensitivity=1.0, utility_fn=lambda s: -s**2,
        )
        assert abs(sigma - 1.0) < 1e-12
        assert abs(util - (-1.0)) < 1e-12

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            GaussianOptimizer.sigma_for_rho(-1.0)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            GaussianOptimizer.rho_for_sigma(0.0)


# =============================================================================
# DiscreteGaussianSynthesis Tests
# =============================================================================


class TestDiscreteGaussianSynthesis:
    """Tests for DiscreteGaussianSynthesis zCDP satisfaction."""

    def test_zcdp_cost_large_sigma(self):
        """For σ² ≥ 1, cost ≈ Δ²/(2σ²)."""
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0, sensitivity=1)
        rho = dg.zcdp_cost()
        expected = 1.0 / 8.0
        assert abs(rho - expected) < 1e-10

    def test_zcdp_cost_small_sigma(self):
        """For small σ², cost includes correction."""
        dg = DiscreteGaussianSynthesis(sigma_sq=0.5, sensitivity=1)
        rho = dg.zcdp_cost()
        rho_continuous = 1.0 / (2.0 * 0.5)
        assert rho > rho_continuous  # correction increases cost

    def test_from_rho(self):
        dg = DiscreteGaussianSynthesis.from_rho(0.5, sensitivity=1)
        assert dg.sigma_sq == 1.0
        assert dg.sensitivity == 1

    def test_sample_shape(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0)
        samples = dg.sample(size=100, rng=np.random.default_rng(42))
        assert samples.shape == (100,)
        assert samples.dtype == np.int64

    def test_sample_integer(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0)
        samples = dg.sample(size=50, rng=np.random.default_rng(0))
        for s in samples:
            assert s == int(s)

    def test_sample_small_sigma(self):
        """Small σ uses enumeration sampling."""
        dg = DiscreteGaussianSynthesis(sigma_sq=0.5)
        samples = dg.sample(size=30, rng=np.random.default_rng(1))
        assert len(samples) == 30

    def test_pmf_sums_to_one(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0)
        total = sum(dg.pmf(k) for k in range(-50, 51))
        assert abs(total - 1.0) < 0.01

    def test_pmf_symmetric(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0)
        assert abs(dg.pmf(3) - dg.pmf(-3)) < 1e-12

    def test_pmf_peak_at_zero(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0)
        assert dg.pmf(0) > dg.pmf(1)
        assert dg.pmf(0) > dg.pmf(-1)

    def test_to_approx_dp(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=4.0)
        dp = dg.to_approx_dp(delta=1e-5)
        assert dp.epsilon > 0

    def test_invalid_sigma_sq(self):
        with pytest.raises(ValueError):
            DiscreteGaussianSynthesis(sigma_sq=0.0)

    def test_name(self):
        dg = DiscreteGaussianSynthesis(sigma_sq=1.0)
        assert dg.name == "discrete_gaussian"


# =============================================================================
# TruncatedGaussianSynthesis Tests
# =============================================================================


class TestTruncatedGaussianSynthesis:
    """Tests for TruncatedGaussianSynthesis."""

    def test_zcdp_cost_large_bound(self):
        """Large B → cost ≈ continuous Gaussian."""
        tg = TruncatedGaussianSynthesis(sigma=1.0, bound=20.0)
        rho_base = 1.0 / 2.0
        assert abs(tg.zcdp_cost() - rho_base) < 0.01

    def test_zcdp_cost_small_bound_higher(self):
        """Small B increases privacy cost."""
        tg_large = TruncatedGaussianSynthesis(sigma=1.0, bound=10.0)
        tg_small = TruncatedGaussianSynthesis(sigma=1.0, bound=2.0)
        assert tg_small.zcdp_cost() >= tg_large.zcdp_cost() - 1e-10

    def test_from_rho(self):
        tg = TruncatedGaussianSynthesis.from_rho(0.5)
        assert tg.sigma > 0
        assert tg.bound > 0

    def test_sample_within_bounds(self):
        tg = TruncatedGaussianSynthesis(sigma=1.0, bound=3.0)
        samples = tg.sample(size=100, rng=np.random.default_rng(42))
        assert np.all(np.abs(samples) <= 3.0 + 1e-10)

    def test_max_error(self):
        tg = TruncatedGaussianSynthesis(sigma=1.0, bound=5.0)
        assert tg.max_error() == 5.0

    def test_expected_error_positive(self):
        tg = TruncatedGaussianSynthesis(sigma=1.0, bound=5.0)
        assert tg.expected_error() > 0

    def test_name(self):
        tg = TruncatedGaussianSynthesis(sigma=1.0, bound=3.0)
        assert tg.name == "truncated_gaussian"


# =============================================================================
# MultiQuerySynthesis Tests
# =============================================================================


class TestMultiQuerySynthesis:
    """Tests for MultiQuerySynthesis budget splitting."""

    def test_equal_allocation(self):
        sens = np.array([1.0, 1.0, 1.0])
        mq = MultiQuerySynthesis(sens, total_rho=3.0)
        rhos, sigmas = mq.equal_allocation()
        np.testing.assert_allclose(rhos, 1.0, atol=1e-12)
        assert len(sigmas) == 3

    def test_equal_allocation_sums_to_total(self):
        sens = np.array([1.0, 2.0])
        mq = MultiQuerySynthesis(sens, total_rho=1.0)
        rhos, _ = mq.equal_allocation()
        np.testing.assert_allclose(rhos.sum(), 1.0, atol=1e-12)

    def test_minimize_total_mse(self):
        sens = np.array([1.0, 2.0])
        mq = MultiQuerySynthesis(sens, total_rho=1.0)
        rhos, sigmas = mq.minimize_total_mse()
        np.testing.assert_allclose(rhos.sum(), 1.0, atol=1e-10)
        # ρ_i ∝ Δ_i
        assert rhos[1] / rhos[0] == pytest.approx(2.0, rel=1e-6)

    def test_minimize_max_mse(self):
        sens = np.array([1.0, 2.0])
        mq = MultiQuerySynthesis(sens, total_rho=1.0)
        rhos, sigmas = mq.minimize_max_mse()
        np.testing.assert_allclose(rhos.sum(), 1.0, atol=1e-10)
        # MSE should be equalized: Δ²/(2ρ) equal
        mses = sens**2 / (2.0 * rhos)
        np.testing.assert_allclose(mses[0], mses[1], atol=1e-10)

    def test_minimize_weighted_mse(self):
        sens = np.array([1.0, 1.0])
        mq = MultiQuerySynthesis(sens, total_rho=1.0)
        weights = np.array([4.0, 1.0])
        rhos, _ = mq.minimize_weighted_mse(weights)
        np.testing.assert_allclose(rhos.sum(), 1.0, atol=1e-10)
        # Higher weight → more budget
        assert rhos[0] > rhos[1]

    def test_custom_objective(self):
        sens = np.array([1.0, 1.0])
        mq = MultiQuerySynthesis(sens, total_rho=1.0)
        rhos, sigmas = mq.custom_objective(lambda s: np.sum(s**2))
        np.testing.assert_allclose(rhos.sum(), 1.0, atol=1e-6)

    def test_summary(self):
        sens = np.array([1.0, 2.0])
        mq = MultiQuerySynthesis(sens, total_rho=1.0)
        rhos, sigmas = mq.equal_allocation()
        s = mq.summary(rhos, sigmas)
        assert s["num_queries"] == 2
        assert abs(s["total_rho"] - 1.0) < 1e-10

    def test_invalid_sensitivities(self):
        with pytest.raises(ValueError):
            MultiQuerySynthesis(np.array([0.0, 1.0]), total_rho=1.0)

    def test_invalid_total_rho(self):
        with pytest.raises(ValueError):
            MultiQuerySynthesis(np.array([1.0]), total_rho=-1.0)


# =============================================================================
# BudgetAllocation Tests
# =============================================================================


class TestBudgetAllocation:
    """Tests for BudgetAllocation optimality."""

    def test_equal(self):
        alloc = BudgetAllocation.equal(1.0, k=4)
        np.testing.assert_allclose(alloc, 0.25, atol=1e-12)
        assert len(alloc) == 4

    def test_proportional(self):
        alloc = BudgetAllocation.proportional(1.0, np.array([1.0, 3.0]))
        np.testing.assert_allclose(alloc.sum(), 1.0, atol=1e-12)
        assert alloc[1] / alloc[0] == pytest.approx(3.0)

    def test_geometric(self):
        alloc = BudgetAllocation.geometric(1.0, k=3, ratio=0.5)
        np.testing.assert_allclose(alloc.sum(), 1.0, atol=1e-10)
        assert alloc[0] > alloc[1] > alloc[2]

    def test_geometric_ratio(self):
        alloc = BudgetAllocation.geometric(1.0, k=3, ratio=0.5)
        assert alloc[1] / alloc[0] == pytest.approx(0.5, rel=1e-6)
        assert alloc[2] / alloc[1] == pytest.approx(0.5, rel=1e-6)

    def test_minimize_mse_gaussian(self):
        sens = np.array([1.0, 2.0, 3.0])
        alloc = BudgetAllocation.minimize_mse_gaussian(1.0, sens)
        np.testing.assert_allclose(alloc.sum(), 1.0, atol=1e-12)
        # ρ_i ∝ Δ_i
        assert alloc[1] / alloc[0] == pytest.approx(2.0, rel=1e-6)

    def test_svt_allocation(self):
        rho_t, rho_a = BudgetAllocation.svt_allocation(1.0, num_threshold=10, num_above=5)
        assert rho_t > 0
        assert rho_a > 0
        assert abs(rho_t + 5 * rho_a - 1.0) < 1e-10

    def test_validate_valid(self):
        alloc = np.array([0.5, 0.3, 0.2])
        assert BudgetAllocation.validate_allocation(alloc, 1.0)

    def test_validate_invalid_sum(self):
        alloc = np.array([0.5, 0.3, 0.1])
        assert not BudgetAllocation.validate_allocation(alloc, 1.0)

    def test_validate_negative(self):
        alloc = np.array([-0.1, 0.6, 0.5])
        assert not BudgetAllocation.validate_allocation(alloc, 1.0)

    def test_invalid_total_rho(self):
        with pytest.raises(ValueError):
            BudgetAllocation.equal(-1.0, k=3)

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            BudgetAllocation.geometric(1.0, k=3, ratio=1.5)


# =============================================================================
# ZCDPSynthesizer Tests
# =============================================================================


class TestZCDPSynthesizer:
    """Tests for ZCDPSynthesizer mechanism production."""

    def test_synthesize_gaussian(self):
        synth = ZCDPSynthesizer()
        qv = np.array([0.0, 0.5, 1.0])
        result = synth.synthesize_gaussian(qv, rho=0.5, sensitivity=1.0, k=20)
        assert result["mechanism"].shape == (3, 20)
        assert abs(result["rho"] - 0.5) < 1e-12

    def test_mechanism_row_stochastic(self):
        synth = ZCDPSynthesizer()
        qv = np.array([0.0, 1.0])
        result = synth.synthesize_gaussian(qv, rho=1.0, k=50)
        M = result["mechanism"]
        for i in range(2):
            np.testing.assert_allclose(M[i].sum(), 1.0, atol=1e-6)

    def test_mechanism_nonneg(self):
        synth = ZCDPSynthesizer()
        qv = np.array([0.0, 0.5, 1.0])
        result = synth.synthesize_gaussian(qv, rho=0.5, k=30)
        assert np.all(result["mechanism"] >= -1e-12)

    def test_sigma_consistent(self):
        synth = ZCDPSynthesizer()
        rho = 0.5
        result = synth.synthesize_gaussian(np.array([0.0]), rho=rho)
        expected_sigma = 1.0 / math.sqrt(2.0 * rho)
        assert abs(result["sigma"] - expected_sigma) < 1e-10

    def test_synthesize_discrete_gaussian(self):
        synth = ZCDPSynthesizer()
        qv = np.array([0, 5, 10], dtype=np.int64)
        result = synth.synthesize_discrete_gaussian(qv, rho=0.5, sensitivity=1)
        M = result["mechanism"]
        assert M.shape[0] == 3
        for i in range(3):
            np.testing.assert_allclose(M[i].sum(), 1.0, atol=1e-6)

    def test_synthesize_for_composition(self):
        synth = ZCDPSynthesizer()
        qvs = [np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])]
        sens = np.array([1.0, 2.0])
        results = synth.synthesize_for_composition(qvs, sens, total_rho=1.0)
        assert len(results) == 2
        for r in results:
            assert r["mechanism"].shape[0] > 0

    def test_invalid_rho(self):
        synth = ZCDPSynthesizer()
        with pytest.raises(ValueError):
            synth.synthesize_gaussian(np.array([0.0]), rho=-1.0)
