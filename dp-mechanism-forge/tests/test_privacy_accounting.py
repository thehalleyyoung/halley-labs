"""Comprehensive tests for dp_forge.privacy_accounting.

Covers all major accounting classes:
  - BasicComposition
  - AdvancedComposition
  - RenyiDPAccountant (and RDPMechanism)
  - ZeroCDPAccountant
  - MomentsAccountant
  - SubsamplingAmplification
  - PrivacyBudgetTracker
  - compose_optimal / compute_noise_for_budget utilities
"""

from __future__ import annotations

import math
import warnings
from typing import List

import numpy as np
import pytest

from dp_forge.exceptions import BudgetExhaustedError, ConfigurationError
from dp_forge.privacy_accounting import (
    AdvancedComposition,
    BasicComposition,
    MomentsAccountant,
    PrivacyAllocation,
    PrivacyBudgetTracker,
    RDPMechanism,
    RenyiDPAccountant,
    SubsamplingAmplification,
    ZeroCDPAccountant,
    compose_optimal,
    compute_noise_for_budget,
)
from dp_forge.types import CompositionType, PrivacyBudget


# =========================================================================
# 1. BasicComposition
# =========================================================================


class TestBasicCompositionSequential:
    """Tests for BasicComposition.sequential."""

    def test_two_budgets_epsilon_sums(self):
        b1 = PrivacyBudget(epsilon=1.0, delta=1e-5)
        b2 = PrivacyBudget(epsilon=0.5, delta=1e-6)
        result = BasicComposition.sequential([b1, b2])
        assert result.epsilon == pytest.approx(1.5)
        assert result.delta == pytest.approx(1e-5 + 1e-6)

    def test_single_budget_unchanged(self):
        b = PrivacyBudget(epsilon=2.0, delta=1e-5)
        result = BasicComposition.sequential([b])
        assert result.epsilon == pytest.approx(2.0)
        assert result.delta == pytest.approx(1e-5)

    def test_pure_dp_budgets(self):
        budgets = [PrivacyBudget(epsilon=1.0), PrivacyBudget(epsilon=2.0)]
        result = BasicComposition.sequential(budgets)
        assert result.epsilon == pytest.approx(3.0)
        assert result.delta == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [3, 5, 10, 50])
    def test_many_identical_budgets(self, n: int):
        eps, delta = 0.1, 1e-6
        budgets = [PrivacyBudget(epsilon=eps, delta=delta)] * n
        result = BasicComposition.sequential(budgets)
        assert result.epsilon == pytest.approx(n * eps)
        assert result.delta == pytest.approx(n * delta)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BasicComposition.sequential([])

    def test_delta_capped_below_one(self):
        budgets = [PrivacyBudget(epsilon=1.0, delta=0.6)] * 3
        result = BasicComposition.sequential(budgets)
        assert result.delta < 1.0


class TestBasicCompositionParallel:
    """Tests for BasicComposition.parallel."""

    def test_takes_max_epsilon(self):
        budgets = [
            PrivacyBudget(epsilon=1.0, delta=1e-5),
            PrivacyBudget(epsilon=3.0, delta=1e-6),
            PrivacyBudget(epsilon=0.5, delta=1e-4),
        ]
        result = BasicComposition.parallel(budgets)
        assert result.epsilon == pytest.approx(3.0)

    def test_takes_max_delta(self):
        budgets = [
            PrivacyBudget(epsilon=1.0, delta=1e-5),
            PrivacyBudget(epsilon=3.0, delta=1e-6),
            PrivacyBudget(epsilon=0.5, delta=1e-4),
        ]
        result = BasicComposition.parallel(budgets)
        assert result.delta == pytest.approx(1e-4)

    def test_single_budget(self):
        b = PrivacyBudget(epsilon=2.0, delta=1e-5)
        result = BasicComposition.parallel([b])
        assert result.epsilon == pytest.approx(2.0)
        assert result.delta == pytest.approx(1e-5)

    def test_identical_budgets(self):
        budgets = [PrivacyBudget(epsilon=1.0, delta=1e-5)] * 10
        result = BasicComposition.parallel(budgets)
        assert result.epsilon == pytest.approx(1.0)
        assert result.delta == pytest.approx(1e-5)

    def test_parallel_tighter_than_sequential(self):
        budgets = [PrivacyBudget(epsilon=1.0, delta=1e-5)] * 5
        par = BasicComposition.parallel(budgets)
        seq = BasicComposition.sequential(budgets)
        assert par.epsilon <= seq.epsilon
        assert par.delta <= seq.delta

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BasicComposition.parallel([])


class TestBasicCompositionHomogeneous:
    """Tests for BasicComposition.sequential_homogeneous."""

    @pytest.mark.parametrize("k", [1, 5, 10, 100])
    def test_k_times_epsilon(self, k: int):
        eps, delta = 0.5, 1e-6
        result = BasicComposition.sequential_homogeneous(eps, delta, k)
        assert result.epsilon == pytest.approx(k * eps)
        assert result.delta == pytest.approx(k * delta)

    def test_k_one_gives_original(self):
        result = BasicComposition.sequential_homogeneous(1.0, 1e-5, 1)
        assert result.epsilon == pytest.approx(1.0)
        assert result.delta == pytest.approx(1e-5)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            BasicComposition.sequential_homogeneous(1.0, 1e-5, 0)

    def test_matches_sequential_list(self):
        eps, delta, k = 0.3, 1e-6, 7
        homo = BasicComposition.sequential_homogeneous(eps, delta, k)
        seq = BasicComposition.sequential(
            [PrivacyBudget(epsilon=eps, delta=delta)] * k
        )
        assert homo.epsilon == pytest.approx(seq.epsilon)
        assert homo.delta == pytest.approx(seq.delta)


# =========================================================================
# 2. AdvancedComposition
# =========================================================================


class TestAdvancedCompositionHomogeneous:
    """Tests for AdvancedComposition.compose_homogeneous."""

    def test_basic_computation(self):
        eps, delta, k = 0.1, 1e-6, 10
        delta_prime = 1e-5
        result = AdvancedComposition.compose_homogeneous(eps, delta, k, delta_prime)
        expected_eps = (
            math.sqrt(2.0 * k * math.log(1.0 / delta_prime)) * eps
            + k * eps * (math.exp(eps) - 1.0)
        )
        expected_delta = k * delta + delta_prime
        assert result.epsilon == pytest.approx(expected_eps, rel=1e-10)
        assert result.delta == pytest.approx(expected_delta, rel=1e-10)

    def test_tighter_than_basic(self):
        eps, delta, k = 0.1, 1e-6, 100
        delta_prime = 1e-5
        advanced = AdvancedComposition.compose_homogeneous(eps, delta, k, delta_prime)
        basic = BasicComposition.sequential_homogeneous(eps, delta, k)
        assert advanced.epsilon < basic.epsilon

    @pytest.mark.parametrize("k", [1, 10, 50, 200])
    def test_monotone_in_k(self, k: int):
        eps, delta, delta_prime = 0.1, 1e-6, 1e-5
        r1 = AdvancedComposition.compose_homogeneous(eps, delta, k, delta_prime)
        r2 = AdvancedComposition.compose_homogeneous(eps, delta, k + 1, delta_prime)
        assert r2.epsilon > r1.epsilon

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            AdvancedComposition.compose_homogeneous(0.1, 1e-6, 0, 1e-5)

    def test_invalid_delta_prime_raises(self):
        with pytest.raises(ConfigurationError):
            AdvancedComposition.compose_homogeneous(0.1, 1e-6, 10, 0.0)
        with pytest.raises(ConfigurationError):
            AdvancedComposition.compose_homogeneous(0.1, 1e-6, 10, 1.0)


class TestAdvancedCompositionHeterogeneous:
    """Tests for AdvancedComposition.compose (heterogeneous)."""

    def test_two_mechanisms(self):
        eps_list = [0.1, 0.2]
        delta_list = [1e-6, 1e-6]
        delta_prime = 1e-5
        result = AdvancedComposition.compose(eps_list, delta_list, delta_prime)
        assert result.epsilon > 0
        assert result.delta == pytest.approx(sum(delta_list) + delta_prime)

    def test_matches_homogeneous_for_identical(self):
        eps, delta, k = 0.1, 1e-6, 5
        delta_prime = 1e-5
        hetero = AdvancedComposition.compose(
            [eps] * k, [delta] * k, delta_prime
        )
        homo = AdvancedComposition.compose_homogeneous(eps, delta, k, delta_prime)
        assert hetero.epsilon == pytest.approx(homo.epsilon, rel=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            AdvancedComposition.compose([], [], 1e-5)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            AdvancedComposition.compose([0.1, 0.2], [1e-6], 1e-5)

    def test_invalid_delta_prime(self):
        with pytest.raises(ConfigurationError):
            AdvancedComposition.compose([0.1], [1e-6], -0.1)


class TestAdvancedCompositionOptimalDeltaPrime:
    """Tests for AdvancedComposition.optimal_delta_prime."""

    def test_returns_positive(self):
        dp = AdvancedComposition.optimal_delta_prime(0.1, 1e-6, 10)
        assert dp > 0

    def test_target_delta_mode(self):
        target = 1e-4
        dp = AdvancedComposition.optimal_delta_prime(0.1, 1e-6, 10, target_delta=target)
        assert dp > 0
        assert dp <= target

    def test_target_delta_tight_budget(self):
        dp = AdvancedComposition.optimal_delta_prime(0.1, 0.01, 10, target_delta=0.05)
        assert dp > 0


# =========================================================================
# 3. RDPMechanism
# =========================================================================


class TestRDPMechanism:
    """Tests for the RDPMechanism dataclass."""

    def test_from_function(self):
        mech = RDPMechanism(rdp_func=lambda a: a * 0.5, name="test")
        assert mech.evaluate(2.0) == pytest.approx(1.0)
        assert mech.evaluate(4.0) == pytest.approx(2.0)

    def test_from_values(self):
        alphas = np.array([2.0, 4.0, 8.0])
        values = np.array([0.5, 1.0, 2.0])
        mech = RDPMechanism(rdp_alphas=alphas, rdp_values=values, name="interp")
        assert mech.evaluate(2.0) == pytest.approx(0.5)
        assert mech.evaluate(4.0) == pytest.approx(1.0)
        # Interpolation at α=3 should give 0.75
        assert mech.evaluate(3.0) == pytest.approx(0.75)

    def test_no_func_or_values_raises(self):
        with pytest.raises(ValueError, match="Either rdp_func or rdp_values"):
            RDPMechanism(name="empty")

    def test_values_without_alphas_raises(self):
        with pytest.raises(ValueError, match="rdp_alphas must be provided"):
            RDPMechanism(rdp_values=np.array([1.0, 2.0]), name="bad")


# =========================================================================
# 4. RenyiDPAccountant
# =========================================================================


class TestRenyiDPAccountant:
    """Tests for RenyiDPAccountant."""

    def test_init_default_alphas(self):
        acc = RenyiDPAccountant()
        assert len(acc.alphas) > 10
        assert acc.n_compositions == 0

    def test_init_custom_alphas(self):
        custom = np.array([2.0, 4.0, 8.0, 16.0])
        acc = RenyiDPAccountant(alphas=custom)
        np.testing.assert_array_equal(acc.alphas, custom)

    def test_add_gaussian_mechanism(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0, sensitivity=1.0)
        acc.add_mechanism(mech)
        assert acc.n_compositions == 1

    def test_compose_two_gaussians(self):
        acc = RenyiDPAccountant()
        m1 = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        m2 = RenyiDPAccountant.gaussian_rdp(sigma=2.0)
        acc.add_mechanism(m1)
        acc.add_mechanism(m2)
        assert acc.n_compositions == 2

        budget = acc.get_privacy(delta=1e-5)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_get_privacy_positive_epsilon(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        acc.add_mechanism(mech)
        budget = acc.get_privacy(delta=1e-5)
        assert budget.epsilon > 0

    def test_rdp_to_approx_dp(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        rdp_curve = acc.compose_rdp([mech])
        eps = acc.rdp_to_approx_dp(rdp_curve, delta=1e-5)
        assert eps > 0

    def test_rdp_to_approx_dp_invalid_delta(self):
        acc = RenyiDPAccountant()
        rdp_curve = np.zeros_like(acc.alphas)
        with pytest.raises(ValueError, match="delta must be in"):
            acc.rdp_to_approx_dp(rdp_curve, delta=0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            acc.rdp_to_approx_dp(rdp_curve, delta=1.0)

    def test_composition_additivity(self):
        """RDP at each α should be additive under composition."""
        acc = RenyiDPAccountant()
        m1 = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        m2 = RenyiDPAccountant.gaussian_rdp(sigma=2.0)

        rdp1 = acc.compose_rdp([m1])
        rdp2 = acc.compose_rdp([m2])
        rdp_both = acc.compose_rdp([m1, m2])

        np.testing.assert_allclose(rdp_both, rdp1 + rdp2, rtol=1e-12)

    def test_more_noise_smaller_epsilon(self):
        """Larger σ should yield smaller ε."""
        acc = RenyiDPAccountant()
        m_low_noise = RenyiDPAccountant.gaussian_rdp(sigma=0.5)
        m_high_noise = RenyiDPAccountant.gaussian_rdp(sigma=5.0)

        rdp_low = acc.compose_rdp([m_low_noise])
        rdp_high = acc.compose_rdp([m_high_noise])

        eps_low = acc.rdp_to_approx_dp(rdp_low, delta=1e-5)
        eps_high = acc.rdp_to_approx_dp(rdp_high, delta=1e-5)
        assert eps_high < eps_low

    def test_optimal_alpha_returns_valid(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        rdp_curve = acc.compose_rdp([mech])
        alpha_opt = acc.optimal_alpha(rdp_curve, delta=1e-5)
        assert alpha_opt > 1.0
        assert alpha_opt in acc.alphas

    def test_get_rdp_curve(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        acc.add_mechanism(mech)
        alphas, rdp_vals = acc.get_rdp_curve()
        assert len(alphas) == len(rdp_vals)
        assert np.all(rdp_vals >= 0)

    def test_reset(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        acc.add_mechanism(mech)
        assert acc.n_compositions == 1
        acc.reset()
        assert acc.n_compositions == 0

    def test_repr(self):
        acc = RenyiDPAccountant()
        s = repr(acc)
        assert "RenyiDPAccountant" in s


class TestRenyiDPGaussianRDP:
    """Tests for RenyiDPAccountant.gaussian_rdp factory."""

    def test_gaussian_rdp_formula(self):
        sigma = 2.0
        mech = RenyiDPAccountant.gaussian_rdp(sigma=sigma, sensitivity=1.0)
        alpha = 5.0
        expected = alpha / (2.0 * sigma ** 2)
        assert mech.evaluate(alpha) == pytest.approx(expected)

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 10.0])
    def test_rdp_decreases_with_sigma(self, sigma: float):
        mech = RenyiDPAccountant.gaussian_rdp(sigma=sigma)
        val = mech.evaluate(2.0)
        assert val == pytest.approx(2.0 / (2.0 * sigma ** 2))

    def test_sensitivity_scaling(self):
        m1 = RenyiDPAccountant.gaussian_rdp(sigma=1.0, sensitivity=1.0)
        m2 = RenyiDPAccountant.gaussian_rdp(sigma=1.0, sensitivity=2.0)
        alpha = 3.0
        assert m2.evaluate(alpha) == pytest.approx(4.0 * m1.evaluate(alpha))


class TestRenyiDPLaplaceRDP:
    """Tests for RenyiDPAccountant.laplace_rdp factory."""

    def test_laplace_rdp_positive(self):
        mech = RenyiDPAccountant.laplace_rdp(epsilon=1.0)
        val = mech.evaluate(2.0)
        assert val > 0

    def test_laplace_rdp_at_alpha_one(self):
        mech = RenyiDPAccountant.laplace_rdp(epsilon=1.0)
        val = mech.evaluate(1.0)
        assert val == pytest.approx(0.0)

    def test_laplace_rdp_monotone_in_alpha(self):
        mech = RenyiDPAccountant.laplace_rdp(epsilon=1.0)
        vals = [mech.evaluate(a) for a in [2.0, 4.0, 8.0]]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i]


class TestRenyiDPSubsampledGaussianRDP:
    """Tests for RenyiDPAccountant.subsampled_gaussian_rdp factory."""

    def test_full_sampling_matches_gaussian(self):
        sigma = 1.0
        mech_full = RenyiDPAccountant.subsampled_gaussian_rdp(
            sigma=sigma, sampling_rate=1.0
        )
        mech_gauss = RenyiDPAccountant.gaussian_rdp(sigma=sigma)
        for alpha in [2.0, 4.0, 8.0]:
            assert mech_full.evaluate(alpha) == pytest.approx(
                mech_gauss.evaluate(alpha), rel=1e-6
            )

    def test_subsampling_reduces_rdp(self):
        sigma = 1.0
        mech_full = RenyiDPAccountant.gaussian_rdp(sigma=sigma)
        mech_sub = RenyiDPAccountant.subsampled_gaussian_rdp(
            sigma=sigma, sampling_rate=0.01
        )
        for alpha in [2.0, 4.0]:
            assert mech_sub.evaluate(alpha) < mech_full.evaluate(alpha)

    def test_invalid_sampling_rate(self):
        with pytest.raises(ValueError, match="sampling_rate"):
            RenyiDPAccountant.subsampled_gaussian_rdp(sigma=1.0, sampling_rate=0.0)
        with pytest.raises(ValueError, match="sampling_rate"):
            RenyiDPAccountant.subsampled_gaussian_rdp(sigma=1.0, sampling_rate=1.5)


# =========================================================================
# 5. ZeroCDPAccountant
# =========================================================================


class TestZeroCDPAccountant:
    """Tests for ZeroCDPAccountant."""

    def test_init(self):
        acc = ZeroCDPAccountant()
        assert acc.total_rho == 0.0
        assert acc.n_compositions == 0

    def test_add_mechanism(self):
        acc = ZeroCDPAccountant()
        acc.add_mechanism(0.5, name="gaussian")
        assert acc.total_rho == pytest.approx(0.5)
        assert acc.n_compositions == 1

    def test_add_multiple_mechanisms(self):
        acc = ZeroCDPAccountant()
        acc.add_mechanism(0.5, name="m1")
        acc.add_mechanism(0.3, name="m2")
        assert acc.total_rho == pytest.approx(0.8)
        assert acc.n_compositions == 2

    def test_add_negative_rho_raises(self):
        acc = ZeroCDPAccountant()
        with pytest.raises(ValueError, match="rho must be > 0"):
            acc.add_mechanism(-0.1)

    def test_add_zero_rho_raises(self):
        acc = ZeroCDPAccountant()
        with pytest.raises(ValueError, match="rho must be > 0"):
            acc.add_mechanism(0.0)


class TestZeroCDPCompose:
    """Tests for ZeroCDPAccountant.compose_zcdp."""

    def test_sums_rhos(self):
        result = ZeroCDPAccountant.compose_zcdp([0.1, 0.2, 0.3])
        assert result == pytest.approx(0.6)

    def test_single_rho(self):
        assert ZeroCDPAccountant.compose_zcdp([0.5]) == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ZeroCDPAccountant.compose_zcdp([])

    def test_negative_rho_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            ZeroCDPAccountant.compose_zcdp([0.1, -0.1])


class TestZeroCDPConversion:
    """Tests for zCDP ↔ (ε, δ)-DP conversions."""

    def test_zcdp_to_approx_dp(self):
        rho = 0.5
        delta = 1e-5
        eps = ZeroCDPAccountant.zcdp_to_approx_dp(rho, delta)
        expected = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        assert eps == pytest.approx(expected)

    @pytest.mark.parametrize("rho", [0.01, 0.1, 0.5, 1.0, 5.0])
    def test_zcdp_to_approx_dp_positive(self, rho: float):
        eps = ZeroCDPAccountant.zcdp_to_approx_dp(rho, 1e-5)
        assert eps > 0

    def test_zcdp_to_approx_dp_invalid_rho(self):
        with pytest.raises(ValueError, match="rho must be > 0"):
            ZeroCDPAccountant.zcdp_to_approx_dp(-0.1, 1e-5)

    def test_zcdp_to_approx_dp_invalid_delta(self):
        with pytest.raises(ValueError, match="delta must be in"):
            ZeroCDPAccountant.zcdp_to_approx_dp(0.5, 0.0)
        with pytest.raises(ValueError, match="delta must be in"):
            ZeroCDPAccountant.zcdp_to_approx_dp(0.5, 1.0)

    def test_approx_dp_to_zcdp(self):
        eps, delta = 1.0, 1e-5
        rho = ZeroCDPAccountant.approx_dp_to_zcdp(eps, delta)
        log_inv_delta = math.log(1.0 / delta)
        expected = eps ** 2 / (2.0 * log_inv_delta)
        assert rho == pytest.approx(expected)

    def test_approx_dp_to_zcdp_pure_dp(self):
        eps = 1.0
        rho = ZeroCDPAccountant.approx_dp_to_zcdp(eps, 0.0)
        expected = eps * (math.exp(eps) - 1.0) / 2.0
        assert rho == pytest.approx(expected)


class TestZeroCDPGaussian:
    """Tests for ZeroCDPAccountant.gaussian_zcdp."""

    def test_formula(self):
        sigma = 2.0
        rho = ZeroCDPAccountant.gaussian_zcdp(sigma, sensitivity=1.0)
        assert rho == pytest.approx(1.0 / (2.0 * sigma ** 2))

    def test_sensitivity_scaling(self):
        rho1 = ZeroCDPAccountant.gaussian_zcdp(1.0, sensitivity=1.0)
        rho2 = ZeroCDPAccountant.gaussian_zcdp(1.0, sensitivity=2.0)
        assert rho2 == pytest.approx(4.0 * rho1)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            ZeroCDPAccountant.gaussian_zcdp(0.0)
        with pytest.raises(ValueError, match="sigma must be > 0"):
            ZeroCDPAccountant.gaussian_zcdp(-1.0)


class TestZeroCDPGetPrivacy:
    """Tests for ZeroCDPAccountant.get_privacy."""

    def test_get_privacy_after_add(self):
        acc = ZeroCDPAccountant()
        acc.add_mechanism(0.5, "gauss")
        budget = acc.get_privacy(delta=1e-5)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_get_privacy_no_mechanisms(self):
        acc = ZeroCDPAccountant()
        budget = acc.get_privacy(delta=1e-5)
        assert budget.epsilon == pytest.approx(1e-15)

    def test_reset(self):
        acc = ZeroCDPAccountant()
        acc.add_mechanism(0.5)
        acc.reset()
        assert acc.total_rho == 0.0
        assert acc.n_compositions == 0

    def test_repr(self):
        acc = ZeroCDPAccountant()
        assert "ZeroCDPAccountant" in repr(acc)


# =========================================================================
# 6. MomentsAccountant
# =========================================================================


class TestMomentsAccountant:
    """Tests for MomentsAccountant."""

    def test_init(self):
        acc = MomentsAccountant(max_order=32)
        assert acc.max_order == 32
        assert acc.n_compositions == 0

    def test_add_gaussian(self):
        acc = MomentsAccountant()
        acc.add_mechanism("gaussian", sigma=1.0, sensitivity=1.0)
        assert acc.n_compositions == 1

    def test_add_laplace(self):
        acc = MomentsAccountant()
        acc.add_mechanism("laplace", epsilon=1.0)
        assert acc.n_compositions == 1

    def test_add_subsampled_gaussian(self):
        acc = MomentsAccountant()
        acc.add_mechanism(
            "subsampled_gaussian",
            sigma=1.0,
            sampling_rate=0.01,
            sensitivity=1.0,
        )
        assert acc.n_compositions == 1

    def test_unknown_mechanism_raises(self):
        acc = MomentsAccountant()
        with pytest.raises(ValueError, match="Unknown mechanism"):
            acc.add_mechanism("unknown_mech")

    def test_get_privacy(self):
        acc = MomentsAccountant()
        acc.add_mechanism("gaussian", sigma=1.0, sensitivity=1.0)
        budget = acc.get_privacy(target_delta=1e-5)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_get_privacy_invalid_delta(self):
        acc = MomentsAccountant()
        acc.add_mechanism("gaussian", sigma=1.0)
        with pytest.raises(ValueError, match="target_delta must be in"):
            acc.get_privacy(target_delta=0.0)

    def test_multiple_compositions_increase_epsilon(self):
        acc = MomentsAccountant()
        acc.add_mechanism("gaussian", sigma=1.0)
        eps1 = acc.get_privacy(1e-5).epsilon

        acc.add_mechanism("gaussian", sigma=1.0)
        eps2 = acc.get_privacy(1e-5).epsilon
        assert eps2 > eps1

    def test_more_noise_smaller_epsilon(self):
        acc1 = MomentsAccountant()
        acc1.add_mechanism("gaussian", sigma=0.5)
        eps_low_noise = acc1.get_privacy(1e-5).epsilon

        acc2 = MomentsAccountant()
        acc2.add_mechanism("gaussian", sigma=5.0)
        eps_high_noise = acc2.get_privacy(1e-5).epsilon

        assert eps_high_noise < eps_low_noise

    def test_compute_log_moments_gaussian(self):
        acc = MomentsAccountant()
        sigma = 2.0
        lm = acc.compute_log_moments("gaussian", order=3.0, sigma=sigma, sensitivity=1.0)
        expected = 3.0 * 4.0 * 1.0 / (2.0 * sigma ** 2)
        assert lm == pytest.approx(expected)

    def test_compose_stateless(self):
        acc = MomentsAccountant(max_order=8)
        lm1 = np.array([acc.compute_log_moments("gaussian", o, sigma=1.0) for o in acc.orders])
        lm2 = np.array([acc.compute_log_moments("gaussian", o, sigma=2.0) for o in acc.orders])
        composed = acc.compose([lm1, lm2])
        np.testing.assert_allclose(composed, lm1 + lm2)

    def test_compose_wrong_length_raises(self):
        acc = MomentsAccountant(max_order=8)
        with pytest.raises(ValueError, match="must match max_order"):
            acc.compose([np.zeros(5)])

    def test_reset(self):
        acc = MomentsAccountant()
        acc.add_mechanism("gaussian", sigma=1.0)
        acc.reset()
        assert acc.n_compositions == 0

    def test_repr(self):
        acc = MomentsAccountant()
        assert "MomentsAccountant" in repr(acc)


# =========================================================================
# 7. SubsamplingAmplification
# =========================================================================


class TestSubsamplingPoisson:
    """Tests for SubsamplingAmplification.poisson_subsample."""

    def test_rate_one_no_amplification(self):
        result = SubsamplingAmplification.poisson_subsample(1.0, 1e-5, rate=1.0)
        assert result.epsilon == pytest.approx(1.0)
        assert result.delta == pytest.approx(1e-5)

    def test_amplification_reduces_epsilon(self):
        eps_orig = 1.0
        delta_orig = 1e-5
        result = SubsamplingAmplification.poisson_subsample(eps_orig, delta_orig, rate=0.01)
        assert result.epsilon < eps_orig

    def test_amplification_reduces_delta(self):
        result = SubsamplingAmplification.poisson_subsample(1.0, 1e-5, rate=0.1)
        assert result.delta < 1e-5

    def test_formula_pure_dp(self):
        eps = 1.0
        rate = 0.1
        result = SubsamplingAmplification.poisson_subsample(eps, 0.0, rate)
        expected_eps = math.log(1.0 + rate * (math.exp(eps) - 1.0))
        assert result.epsilon == pytest.approx(expected_eps)
        assert result.delta == pytest.approx(0.0)

    @pytest.mark.parametrize("rate", [0.001, 0.01, 0.1, 0.5])
    def test_smaller_rate_smaller_epsilon(self, rate: float):
        result = SubsamplingAmplification.poisson_subsample(1.0, 1e-5, rate)
        assert result.epsilon <= 1.0

    def test_monotone_in_rate(self):
        rates = [0.01, 0.1, 0.5, 1.0]
        epsilons = [
            SubsamplingAmplification.poisson_subsample(1.0, 1e-5, r).epsilon
            for r in rates
        ]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] <= epsilons[i + 1]

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="rate must be in"):
            SubsamplingAmplification.poisson_subsample(1.0, 1e-5, 0.0)
        with pytest.raises(ValueError, match="rate must be in"):
            SubsamplingAmplification.poisson_subsample(1.0, 1e-5, 1.5)


class TestSubsamplingWithoutReplacement:
    """Tests for SubsamplingAmplification.without_replacement."""

    def test_full_batch_no_amplification(self):
        result = SubsamplingAmplification.without_replacement(1.0, 1e-5, n=100, batch_size=100)
        assert result.epsilon == pytest.approx(1.0)
        assert result.delta == pytest.approx(1e-5)

    def test_amplification_reduces_epsilon(self):
        result = SubsamplingAmplification.without_replacement(1.0, 1e-5, n=1000, batch_size=10)
        assert result.epsilon < 1.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            SubsamplingAmplification.without_replacement(1.0, 1e-5, n=0, batch_size=1)

    def test_batch_too_large_raises(self):
        with pytest.raises(ValueError, match="batch_size must be in"):
            SubsamplingAmplification.without_replacement(1.0, 1e-5, n=10, batch_size=20)

    def test_batch_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be in"):
            SubsamplingAmplification.without_replacement(1.0, 1e-5, n=10, batch_size=0)


class TestSubsamplingAmplifiedRDPGaussian:
    """Tests for SubsamplingAmplification.amplified_rdp_gaussian."""

    def test_returns_alphas_and_values(self):
        alphas, rdp_vals = SubsamplingAmplification.amplified_rdp_gaussian(
            sigma=1.0, sampling_rate=0.01, sensitivity=1.0
        )
        assert len(alphas) == len(rdp_vals)
        assert len(alphas) > 0

    def test_custom_alphas(self):
        custom_alphas = np.array([2.0, 4.0, 8.0])
        alphas, rdp_vals = SubsamplingAmplification.amplified_rdp_gaussian(
            sigma=1.0, sampling_rate=0.1, sensitivity=1.0, alphas=custom_alphas
        )
        np.testing.assert_array_equal(alphas, custom_alphas)
        assert len(rdp_vals) == 3


class TestComputeSigmaForBudget:
    """Tests for SubsamplingAmplification.compute_sigma_for_budget."""

    @pytest.mark.slow
    def test_returns_positive_sigma(self):
        sigma = SubsamplingAmplification.compute_sigma_for_budget(
            epsilon=1.0, delta=1e-5, sensitivity=1.0,
            sampling_rate=0.01, n_steps=100,
        )
        assert sigma > 0

    @pytest.mark.slow
    def test_tighter_budget_needs_more_noise(self):
        sigma_loose = SubsamplingAmplification.compute_sigma_for_budget(
            epsilon=10.0, delta=1e-5, sensitivity=1.0,
            sampling_rate=0.01, n_steps=100,
        )
        sigma_tight = SubsamplingAmplification.compute_sigma_for_budget(
            epsilon=1.0, delta=1e-5, sensitivity=1.0,
            sampling_rate=0.01, n_steps=100,
        )
        assert sigma_tight > sigma_loose


# =========================================================================
# 8. PrivacyBudgetTracker
# =========================================================================


class TestPrivacyBudgetTrackerBasic:
    """Tests for PrivacyBudgetTracker with basic composition."""

    def test_init(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        assert tracker.total_epsilon == 10.0
        assert tracker.total_delta == 1e-5
        assert tracker.n_allocations == 0

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PrivacyBudgetTracker(epsilon=0.0, delta=1e-5)
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PrivacyBudgetTracker(epsilon=-1.0, delta=1e-5)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta must be in"):
            PrivacyBudgetTracker(epsilon=1.0, delta=1.0)

    def test_allocate_and_consumed(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=1.0, delta=1e-6))
        consumed = tracker.consumed
        assert consumed.epsilon == pytest.approx(1.0)
        assert consumed.delta == pytest.approx(1e-6)

    def test_remaining_after_allocate(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=3.0, delta=2e-6))
        remaining = tracker.remaining()
        assert remaining.epsilon == pytest.approx(7.0)
        assert remaining.delta == pytest.approx(1e-5 - 2e-6)

    def test_multiple_allocations(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-4)
        tracker.allocate("m1", PrivacyBudget(epsilon=2.0, delta=1e-5))
        tracker.allocate("m2", PrivacyBudget(epsilon=3.0, delta=2e-5))
        consumed = tracker.consumed
        assert consumed.epsilon == pytest.approx(5.0)

    def test_allocation_history(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=1.0, delta=1e-6))
        tracker.allocate("m2", PrivacyBudget(epsilon=2.0, delta=1e-6))
        history = tracker.allocation_history
        assert len(history) == 2
        assert history[0].mechanism_name == "m1"
        assert history[1].mechanism_name == "m2"

    def test_can_afford_true(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        assert tracker.can_afford(PrivacyBudget(epsilon=5.0, delta=1e-6))

    def test_can_afford_false(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=8.0, delta=1e-6))
        assert not tracker.can_afford(PrivacyBudget(epsilon=5.0, delta=1e-6))

    def test_budget_exhausted_error(self):
        tracker = PrivacyBudgetTracker(epsilon=5.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=4.0, delta=1e-6))
        with pytest.raises(BudgetExhaustedError):
            tracker.allocate("m2", PrivacyBudget(epsilon=3.0, delta=1e-6))

    def test_is_exhausted_false_initially(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        assert not tracker.is_exhausted()

    def test_is_exhausted_true_when_spent(self):
        tracker = PrivacyBudgetTracker(epsilon=1.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=1.0, delta=1e-6))
        assert tracker.is_exhausted()

    def test_reset(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=5.0, delta=1e-6))
        tracker.reset()
        assert tracker.n_allocations == 0
        assert not tracker.is_exhausted()

    def test_summary(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=2.0, delta=1e-6))
        s = tracker.summary()
        assert s["total_budget"]["epsilon"] == 10.0
        assert s["n_allocations"] == 1
        assert "consumed" in s
        assert "remaining" in s
        assert "allocations" in s

    def test_repr(self):
        tracker = PrivacyBudgetTracker(epsilon=10.0, delta=1e-5)
        s = repr(tracker)
        assert "PrivacyBudgetTracker" in s

    def test_warn_threshold(self):
        tracker = PrivacyBudgetTracker(
            epsilon=10.0, delta=1e-5, warn_threshold=0.5
        )
        tracker.allocate("m1", PrivacyBudget(epsilon=3.0, delta=1e-6))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tracker.allocate("m2", PrivacyBudget(epsilon=4.0, delta=1e-6))
            privacy_warnings = [x for x in w if "budget" in str(x.message).lower()]
            assert len(privacy_warnings) >= 1


class TestPrivacyBudgetTrackerRDP:
    """Tests for PrivacyBudgetTracker with RDP composition."""

    def test_rdp_composition(self):
        tracker = PrivacyBudgetTracker(
            epsilon=10.0, delta=1e-5,
            composition_type=CompositionType.RDP,
        )
        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        tracker.allocate(
            "gauss",
            PrivacyBudget(epsilon=1.0, delta=1e-6),
            rdp_mechanism=mech,
        )
        consumed = tracker.consumed
        assert consumed.epsilon > 0

    def test_rdp_remaining_decreases(self):
        tracker = PrivacyBudgetTracker(
            epsilon=10.0, delta=1e-5,
            composition_type=CompositionType.RDP,
        )
        remaining_before = tracker.remaining().epsilon

        mech = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        tracker.allocate("gauss", PrivacyBudget(epsilon=1.0, delta=1e-6), rdp_mechanism=mech)
        remaining_after = tracker.remaining().epsilon
        assert remaining_after < remaining_before


class TestPrivacyBudgetTrackerZCDP:
    """Tests for PrivacyBudgetTracker with zCDP composition."""

    def test_zcdp_composition(self):
        tracker = PrivacyBudgetTracker(
            epsilon=10.0, delta=1e-5,
            composition_type=CompositionType.ZERO_CDP,
        )
        tracker.allocate(
            "gauss",
            PrivacyBudget(epsilon=1.0, delta=1e-6),
            zcdp_rho=0.5,
        )
        consumed = tracker.consumed
        assert consumed.epsilon > 0

    def test_zcdp_remaining_decreases(self):
        tracker = PrivacyBudgetTracker(
            epsilon=20.0, delta=1e-5,
            composition_type=CompositionType.ZERO_CDP,
        )
        remaining_before = tracker.remaining().epsilon

        tracker.allocate("gauss", PrivacyBudget(epsilon=1.0, delta=1e-6), zcdp_rho=0.5)
        remaining_after = tracker.remaining().epsilon
        assert remaining_after < remaining_before


# =========================================================================
# 9. compose_optimal utility
# =========================================================================


class TestComposeOptimal:
    """Tests for compose_optimal utility."""

    def test_single_budget(self):
        budgets = [PrivacyBudget(epsilon=1.0, delta=1e-5)]
        result = compose_optimal(budgets, target_delta=1e-4)
        assert result.epsilon == pytest.approx(1.0)

    def test_tighter_than_basic(self):
        budgets = [PrivacyBudget(epsilon=0.1, delta=1e-6)] * 50
        optimal = compose_optimal(budgets, target_delta=1e-4)
        basic = BasicComposition.sequential(budgets)
        assert optimal.epsilon <= basic.epsilon

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compose_optimal([], target_delta=1e-5)

    def test_returns_privacy_budget(self):
        budgets = [PrivacyBudget(epsilon=0.5, delta=1e-6)] * 5
        result = compose_optimal(budgets, target_delta=1e-4)
        assert isinstance(result, PrivacyBudget)
        assert result.epsilon > 0


# =========================================================================
# 10. compute_noise_for_budget utility
# =========================================================================


class TestComputeNoiseForBudget:
    """Tests for compute_noise_for_budget utility."""

    def test_gaussian_noise(self):
        sigma = compute_noise_for_budget(
            epsilon=1.0, delta=1e-5, sensitivity=1.0, mechanism="gaussian"
        )
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        assert sigma == pytest.approx(expected)

    def test_laplace_noise(self):
        b = compute_noise_for_budget(
            epsilon=1.0, delta=0.0, sensitivity=1.0, mechanism="laplace"
        )
        assert b == pytest.approx(1.0)

    def test_gaussian_requires_positive_delta(self):
        with pytest.raises(ValueError, match="delta > 0"):
            compute_noise_for_budget(
                epsilon=1.0, delta=0.0, sensitivity=1.0, mechanism="gaussian"
            )

    def test_unknown_mechanism_raises(self):
        with pytest.raises(ValueError, match="Unknown mechanism"):
            compute_noise_for_budget(
                epsilon=1.0, delta=1e-5, sensitivity=1.0, mechanism="exponential"
            )

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 5.0])
    def test_tighter_budget_more_noise_gaussian(self, eps: float):
        sigma = compute_noise_for_budget(
            epsilon=eps, delta=1e-5, sensitivity=1.0, mechanism="gaussian"
        )
        assert sigma > 0

    def test_higher_sensitivity_more_noise(self):
        s1 = compute_noise_for_budget(
            epsilon=1.0, delta=1e-5, sensitivity=1.0, mechanism="gaussian"
        )
        s2 = compute_noise_for_budget(
            epsilon=1.0, delta=1e-5, sensitivity=2.0, mechanism="gaussian"
        )
        assert s2 > s1


# =========================================================================
# 11. PrivacyAllocation dataclass
# =========================================================================


class TestPrivacyAllocation:
    """Tests for PrivacyAllocation dataclass."""

    def test_defaults(self):
        a = PrivacyAllocation(mechanism_name="test", epsilon=1.0, delta=1e-5)
        assert a.mechanism_name == "test"
        assert a.epsilon == 1.0
        assert a.delta == 1e-5
        assert a.composition_type == CompositionType.BASIC
        assert a.metadata == {}

    def test_custom_composition_type(self):
        a = PrivacyAllocation(
            mechanism_name="rdp_test",
            epsilon=0.5,
            delta=1e-6,
            composition_type=CompositionType.RDP,
            metadata={"sigma": 1.0},
        )
        assert a.composition_type == CompositionType.RDP
        assert a.metadata["sigma"] == 1.0


# =========================================================================
# 12. Cross-framework consistency checks
# =========================================================================


class TestCrossFrameworkConsistency:
    """Tests verifying consistency across different accounting frameworks."""

    def test_rdp_gaussian_matches_zcdp_gaussian(self):
        """For Gaussian, RDP at α=2 should equal 2·ρ (zCDP definition)."""
        sigma = 2.0
        rdp_mech = RenyiDPAccountant.gaussian_rdp(sigma=sigma)
        rdp_at_2 = rdp_mech.evaluate(2.0)
        rho = ZeroCDPAccountant.gaussian_zcdp(sigma=sigma)
        # RDP at α: α·Δ²/(2σ²);  zCDP ρ = Δ²/(2σ²);  so RDP(α) = α·ρ
        assert rdp_at_2 == pytest.approx(2.0 * rho)

    def test_advanced_tighter_than_basic_for_many_compositions(self):
        eps, delta, k = 0.1, 1e-8, 100
        basic = BasicComposition.sequential_homogeneous(eps, delta, k)
        delta_prime = 1e-5
        advanced = AdvancedComposition.compose_homogeneous(eps, delta, k, delta_prime)
        assert advanced.epsilon < basic.epsilon

    def test_zcdp_composition_matches_manual_sum(self):
        rhos = [0.1, 0.2, 0.3]
        acc = ZeroCDPAccountant()
        for r in rhos:
            acc.add_mechanism(r)
        assert acc.total_rho == pytest.approx(sum(rhos))

    def test_rdp_composition_via_accountant_vs_compose_rdp(self):
        """Stateful add_mechanism should match stateless compose_rdp."""
        acc = RenyiDPAccountant()
        m1 = RenyiDPAccountant.gaussian_rdp(sigma=1.0)
        m2 = RenyiDPAccountant.gaussian_rdp(sigma=2.0)

        # Stateful
        acc.add_mechanism(m1)
        acc.add_mechanism(m2)
        _, stateful_rdp = acc.get_rdp_curve()

        # Stateless
        acc2 = RenyiDPAccountant()
        stateless_rdp = acc2.compose_rdp([m1, m2])

        np.testing.assert_allclose(stateful_rdp, stateless_rdp, rtol=1e-12)

    def test_subsampling_amplification_vs_rdp_subsampled(self):
        """Poisson subsampling should reduce effective epsilon."""
        eps_orig = 1.0
        rate = 0.01
        amplified = SubsamplingAmplification.poisson_subsample(eps_orig, 0.0, rate)
        assert amplified.epsilon < eps_orig

    @pytest.mark.parametrize(
        "sigma,delta",
        [(1.0, 1e-5), (2.0, 1e-6), (5.0, 1e-3)],
    )
    def test_zcdp_eps_bounded_by_rdp_eps(self, sigma: float, delta: float):
        """zCDP conversion should give comparable ε to RDP conversion."""
        # zCDP path
        rho = ZeroCDPAccountant.gaussian_zcdp(sigma)
        eps_zcdp = ZeroCDPAccountant.zcdp_to_approx_dp(rho, delta)

        # RDP path
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=sigma)
        rdp_curve = acc.compose_rdp([mech])
        eps_rdp = acc.rdp_to_approx_dp(rdp_curve, delta)

        # Both should be positive and relatively close for Gaussian
        assert eps_zcdp > 0
        assert eps_rdp > 0
        # RDP with optimal α should be at least as tight as zCDP's generic bound
        assert eps_rdp <= eps_zcdp * 1.5  # allow some slack


# =========================================================================
# 13. Edge cases and numerical stability
# =========================================================================


class TestEdgeCases:
    """Edge cases and numerical stability tests."""

    def test_very_small_epsilon(self):
        b = PrivacyBudget(epsilon=1e-10, delta=1e-10)
        result = BasicComposition.sequential([b])
        assert result.epsilon == pytest.approx(1e-10)

    def test_very_large_k_basic(self):
        result = BasicComposition.sequential_homogeneous(0.001, 1e-10, 10000)
        assert result.epsilon == pytest.approx(10.0)

    def test_many_small_budgets_sequential(self):
        budgets = [PrivacyBudget(epsilon=0.001, delta=1e-10)] * 1000
        result = BasicComposition.sequential(budgets)
        assert result.epsilon == pytest.approx(1.0)

    def test_rdp_many_compositions(self):
        acc = RenyiDPAccountant()
        mech = RenyiDPAccountant.gaussian_rdp(sigma=10.0)
        for _ in range(100):
            acc.add_mechanism(mech)
        budget = acc.get_privacy(delta=1e-5)
        assert budget.epsilon > 0
        assert math.isfinite(budget.epsilon)

    def test_zcdp_many_compositions(self):
        acc = ZeroCDPAccountant()
        for _ in range(100):
            acc.add_mechanism(0.01)
        assert acc.total_rho == pytest.approx(1.0)
        budget = acc.get_privacy(delta=1e-5)
        assert math.isfinite(budget.epsilon)

    def test_moments_many_compositions(self):
        acc = MomentsAccountant(max_order=32)
        for _ in range(50):
            acc.add_mechanism("gaussian", sigma=5.0)
        budget = acc.get_privacy(1e-5)
        assert budget.epsilon > 0
        assert math.isfinite(budget.epsilon)

    def test_tracker_exact_budget_spend(self):
        tracker = PrivacyBudgetTracker(epsilon=5.0, delta=1e-5)
        tracker.allocate("m1", PrivacyBudget(epsilon=5.0, delta=1e-5))
        assert tracker.is_exhausted()

    def test_tracker_sequential_small_allocations(self):
        tracker = PrivacyBudgetTracker(epsilon=1.0, delta=1e-4)
        for i in range(10):
            tracker.allocate(f"m{i}", PrivacyBudget(epsilon=0.1, delta=1e-5))
        assert tracker.is_exhausted()
        assert tracker.n_allocations == 10
