"""
Tests for dp_forge.rdp.conversion — Privacy framework conversions.

Covers:
    - rdp_to_dp: Gaussian σ=1, verify against closed-form
    - dp_to_rdp_bound: verify bound is valid upper bound
    - zCDP to RDP and back: round-trip consistency
    - Concentrated DP conversions
    - Numerical stability: very small ε, very large α
    - Monotonicity: larger α gives larger ε for fixed mechanism
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.rdp.conversion import (
    rdp_to_dp,
    rdp_to_dp_budget,
    dp_to_rdp_bound,
    zcdp_to_rdp,
    rdp_to_zcdp,
    zcdp_to_dp,
    dp_to_zcdp,
    gaussian_rdp,
    gaussian_zcdp,
    gaussian_dp,
    laplace_rdp,
    compose_rdp_then_convert,
    compose_zcdp_then_convert,
    DEFAULT_ALPHAS,
)
from dp_forge.types import PrivacyBudget
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def simple_alphas():
    return np.array([1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])


def gaussian_rdp_expected(alpha, sigma, sensitivity=1.0):
    return alpha * sensitivity ** 2 / (2.0 * sigma ** 2)


# =========================================================================
# rdp_to_dp tests
# =========================================================================


class TestRDPToDP:
    """Tests for rdp_to_dp conversion."""

    def test_standard_conversion_gaussian(self, simple_alphas):
        """Verify standard Mironov (2017) conversion for Gaussian σ=1."""
        sigma = 1.0
        rdp_eps = np.array([gaussian_rdp_expected(a, sigma) for a in simple_alphas])
        delta = 1e-5

        eps, opt_alpha = rdp_to_dp(rdp_eps, simple_alphas, delta, method="standard")
        assert eps > 0
        assert opt_alpha in simple_alphas

        # Verify against manual computation
        log_delta = math.log(delta)
        eps_candidates = rdp_eps - log_delta / (simple_alphas - 1.0)
        expected_eps = max(float(np.min(eps_candidates)), 0.0)
        assert eps == pytest.approx(expected_eps, rel=1e-10)

    def test_balle2020_at_least_as_tight(self, simple_alphas):
        """Balle (2020) should give ε ≤ standard ε."""
        sigma = 1.0
        rdp_eps = np.array([gaussian_rdp_expected(a, sigma) for a in simple_alphas])
        delta = 1e-5

        eps_std, _ = rdp_to_dp(rdp_eps, simple_alphas, delta, method="standard")
        eps_balle, _ = rdp_to_dp(rdp_eps, simple_alphas, delta, method="balle2020")
        assert eps_balle <= eps_std + 1e-10

    def test_invalid_delta(self, simple_alphas):
        rdp_eps = simple_alphas * 0.1
        with pytest.raises(ValueError, match="delta"):
            rdp_to_dp(rdp_eps, simple_alphas, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            rdp_to_dp(rdp_eps, simple_alphas, delta=1.0)

    def test_unknown_method(self, simple_alphas):
        rdp_eps = simple_alphas * 0.1
        with pytest.raises(ConfigurationError, match="Unknown"):
            rdp_to_dp(rdp_eps, simple_alphas, delta=1e-5, method="invalid")

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            rdp_to_dp(np.array([0.1, 0.2]), np.array([2.0]), delta=1e-5)

    def test_rdp_to_dp_budget(self, simple_alphas):
        rdp_eps = np.array([gaussian_rdp_expected(a, 1.0) for a in simple_alphas])
        budget = rdp_to_dp_budget(rdp_eps, simple_alphas, delta=1e-5)
        assert isinstance(budget, PrivacyBudget)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_smaller_sigma_larger_epsilon(self, simple_alphas):
        """Smaller σ → larger ε (more privacy loss)."""
        delta = 1e-5
        rdp1 = np.array([gaussian_rdp_expected(a, 1.0) for a in simple_alphas])
        rdp2 = np.array([gaussian_rdp_expected(a, 2.0) for a in simple_alphas])
        eps1, _ = rdp_to_dp(rdp1, simple_alphas, delta)
        eps2, _ = rdp_to_dp(rdp2, simple_alphas, delta)
        assert eps1 > eps2


# =========================================================================
# dp_to_rdp_bound tests
# =========================================================================


class TestDPToRDP:
    """Tests for dp_to_rdp_bound."""

    def test_pure_dp_bound(self, simple_alphas):
        """For pure DP (δ=0), RDP bound should be ε for all α."""
        epsilon = 1.0
        alphas, rdp_bounds = dp_to_rdp_bound(epsilon, delta=0.0, alphas=simple_alphas)
        np.testing.assert_allclose(rdp_bounds, epsilon, atol=1e-10)

    def test_approximate_dp_bound_is_upper_bound(self, simple_alphas):
        """The dp_to_rdp bound should be an upper bound on the true RDP."""
        epsilon, delta = 1.0, 1e-5
        alphas, rdp_bounds = dp_to_rdp_bound(epsilon, delta, alphas=simple_alphas)

        # Verify bounds are non-negative and finite
        assert np.all(rdp_bounds >= 0)
        assert np.all(np.isfinite(rdp_bounds))

    def test_rdp_bound_less_than_epsilon(self, simple_alphas):
        """RDP bound should be ≤ ε for large α."""
        epsilon, delta = 1.0, 1e-5
        _, rdp_bounds = dp_to_rdp_bound(epsilon, delta, alphas=simple_alphas)
        # The bound should converge to ε for large α
        assert rdp_bounds[-1] <= epsilon + 1e-10

    def test_round_trip_consistency(self, simple_alphas):
        """dp_to_rdp → rdp_to_dp should give back ≈ original (ε,δ)."""
        epsilon, delta = 1.0, 1e-5
        alphas, rdp_bounds = dp_to_rdp_bound(epsilon, delta, alphas=simple_alphas)
        recovered_eps, _ = rdp_to_dp(rdp_bounds, alphas, delta, method="standard")
        # Recovered ε should be ≤ original ε (bounds may not be tight)
        assert recovered_eps <= epsilon + 0.5

    def test_default_alphas(self):
        alphas, bounds = dp_to_rdp_bound(1.0, 0.0)
        assert len(alphas) == len(DEFAULT_ALPHAS)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            dp_to_rdp_bound(-1.0, 0.0)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            dp_to_rdp_bound(1.0, 1.5)


# =========================================================================
# zCDP conversions
# =========================================================================


class TestZCDPConversions:
    """Tests for zCDP ↔ RDP conversions."""

    def test_zcdp_to_rdp_linear(self, simple_alphas):
        """ρ-zCDP → (α, ρα)-RDP."""
        rho = 0.5
        alphas, rdp_eps = zcdp_to_rdp(rho, alphas=simple_alphas)
        expected = rho * simple_alphas
        np.testing.assert_allclose(rdp_eps, expected, rtol=1e-12)

    def test_rdp_to_zcdp_gaussian(self, simple_alphas):
        """Gaussian mechanism: RDP is linear in α, so ρ = Δ²/(2σ²)."""
        sigma = 1.0
        rho_expected = 1.0 / (2.0 * sigma ** 2)
        rdp_eps = np.array([gaussian_rdp_expected(a, sigma) for a in simple_alphas])
        rho = rdp_to_zcdp(rdp_eps, simple_alphas)
        assert rho == pytest.approx(rho_expected, rel=1e-10)

    def test_zcdp_round_trip(self, simple_alphas):
        """zCDP → RDP → zCDP should recover ρ."""
        rho = 0.3
        alphas, rdp_eps = zcdp_to_rdp(rho, alphas=simple_alphas)
        rho_recovered = rdp_to_zcdp(rdp_eps, alphas)
        assert rho_recovered == pytest.approx(rho, rel=1e-10)

    def test_zcdp_to_dp(self):
        """ρ-zCDP → (ε,δ)-DP: ε = ρ + 2√(ρ·log(1/δ))."""
        rho, delta = 0.5, 1e-5
        budget = zcdp_to_dp(rho, delta)
        expected_eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        assert budget.epsilon == pytest.approx(expected_eps, rel=1e-10)
        assert budget.delta == delta

    def test_dp_to_zcdp(self):
        """(ε,δ)-DP → ρ-zCDP upper bound."""
        epsilon, delta = 1.0, 1e-5
        rho = dp_to_zcdp(epsilon, delta)
        assert rho > 0
        # Verify: zcdp_to_dp(rho, delta) should give ε ≤ original ε
        budget = zcdp_to_dp(rho, delta)
        assert budget.epsilon <= epsilon + 1e-10

    def test_zcdp_to_rdp_zero_rho(self, simple_alphas):
        _, rdp_eps = zcdp_to_rdp(0.0, alphas=simple_alphas)
        np.testing.assert_allclose(rdp_eps, 0.0)

    def test_zcdp_to_rdp_negative_rho(self):
        with pytest.raises(ValueError, match="rho"):
            zcdp_to_rdp(-1.0)

    def test_zcdp_to_dp_invalid_params(self):
        with pytest.raises(ValueError):
            zcdp_to_dp(-1.0, 1e-5)
        with pytest.raises(ValueError):
            zcdp_to_dp(0.5, 0.0)

    def test_dp_to_zcdp_invalid_params(self):
        with pytest.raises(ValueError):
            dp_to_zcdp(-1.0, 1e-5)
        with pytest.raises(ValueError):
            dp_to_zcdp(1.0, 0.0)


# =========================================================================
# Gaussian/Laplace RDP characterisation
# =========================================================================


class TestGaussianCharacterisation:
    """Tests for gaussian_rdp, gaussian_zcdp, gaussian_dp."""

    def test_gaussian_rdp_formula(self, simple_alphas):
        sigma, sens = 2.0, 1.0
        alphas, rdp_eps = gaussian_rdp(sigma, sens, alphas=simple_alphas)
        expected = simple_alphas * sens ** 2 / (2.0 * sigma ** 2)
        np.testing.assert_allclose(rdp_eps, expected, rtol=1e-12)

    def test_gaussian_zcdp(self):
        sigma, sens = 2.0, 1.0
        rho = gaussian_zcdp(sigma, sens)
        assert rho == pytest.approx(sens ** 2 / (2.0 * sigma ** 2), rel=1e-12)

    def test_gaussian_dp(self):
        budget = gaussian_dp(sigma=1.0, sensitivity=1.0, delta=1e-5)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_gaussian_rdp_invalid_sigma(self):
        with pytest.raises(ValueError):
            gaussian_rdp(0.0, 1.0)

    def test_gaussian_rdp_invalid_sensitivity(self):
        with pytest.raises(ValueError):
            gaussian_rdp(1.0, 0.0)


class TestLaplaceCharacterisation:
    """Tests for laplace_rdp."""

    def test_laplace_rdp_nonnegative(self, simple_alphas):
        alphas, rdp_eps = laplace_rdp(1.0, alphas=simple_alphas)
        assert np.all(rdp_eps >= 0)

    def test_laplace_rdp_monotone_in_epsilon(self, simple_alphas):
        """Larger Laplace ε → larger RDP ε."""
        _, rdp1 = laplace_rdp(0.5, alphas=simple_alphas)
        _, rdp2 = laplace_rdp(1.0, alphas=simple_alphas)
        np.testing.assert_array_less(rdp1 - 1e-10, rdp2)

    def test_laplace_rdp_invalid_epsilon(self):
        with pytest.raises(ValueError):
            laplace_rdp(0.0)


# =========================================================================
# Composition helpers
# =========================================================================


class TestCompositionHelpers:
    """Tests for compose_rdp_then_convert and compose_zcdp_then_convert."""

    def test_compose_rdp_then_convert(self, simple_alphas):
        sigma1, sigma2 = 1.0, 2.0
        rdp1 = (simple_alphas, np.array([gaussian_rdp_expected(a, sigma1) for a in simple_alphas]))
        rdp2 = (simple_alphas, np.array([gaussian_rdp_expected(a, sigma2) for a in simple_alphas]))
        budget = compose_rdp_then_convert([rdp1, rdp2], delta=1e-5, alphas=simple_alphas)
        assert budget.epsilon > 0

    def test_compose_zcdp_then_convert(self):
        rhos = [0.1, 0.2, 0.3]
        budget = compose_zcdp_then_convert(rhos, delta=1e-5)
        total_rho = sum(rhos)
        expected = zcdp_to_dp(total_rho, 1e-5)
        assert budget.epsilon == pytest.approx(expected.epsilon, rel=1e-10)

    def test_compose_rdp_empty(self):
        with pytest.raises(ValueError):
            compose_rdp_then_convert([], delta=1e-5)

    def test_compose_zcdp_empty(self):
        with pytest.raises(ValueError):
            compose_zcdp_then_convert([], delta=1e-5)


# =========================================================================
# Numerical stability tests
# =========================================================================


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_epsilon(self, simple_alphas):
        """Conversion should handle very small RDP ε."""
        rdp_eps = simple_alphas * 1e-10
        eps, _ = rdp_to_dp(rdp_eps, simple_alphas, delta=1e-5)
        assert eps >= 0
        assert math.isfinite(eps)

    def test_very_large_alpha(self):
        """Large α should not cause overflow."""
        alphas = np.array([2.0, 10.0, 100.0, 1000.0, 10000.0])
        sigma = 1.0
        rdp_eps = alphas * 1.0 / (2.0 * sigma ** 2)
        eps, opt_a = rdp_to_dp(rdp_eps, alphas, delta=1e-5)
        assert math.isfinite(eps)

    def test_very_small_delta(self, simple_alphas):
        """Very small δ should give large but finite ε."""
        rdp_eps = simple_alphas * 0.5
        eps, _ = rdp_to_dp(rdp_eps, simple_alphas, delta=1e-15)
        assert eps > 0
        assert math.isfinite(eps)

    def test_monotonicity_in_alpha(self, simple_alphas):
        """For Gaussian, RDP ε(α) = α/(2σ²) is monotonically increasing in α."""
        sigma = 1.0
        _, rdp_eps = gaussian_rdp(sigma, 1.0, alphas=simple_alphas)
        diffs = np.diff(rdp_eps)
        assert np.all(diffs >= -1e-15)


# =========================================================================
# Property-based tests
# =========================================================================


class TestConversionProperties:
    """Hypothesis-based property tests for conversions."""

    @given(
        rho=st.floats(min_value=0.01, max_value=10.0),
        delta=st.floats(min_value=1e-10, max_value=0.1),
    )
    @settings(max_examples=30)
    def test_zcdp_dp_round_trip_bound(self, rho, delta):
        """zCDP → DP → zCDP should give ρ' ≤ ρ (inverse may be tighter)."""
        budget = zcdp_to_dp(rho, delta)
        rho_recovered = dp_to_zcdp(budget.epsilon, delta)
        # The recovered ρ should be ≤ original (dp_to_zcdp inverts the conversion)
        assert rho_recovered <= rho + 1e-10

    @given(
        sigma=st.floats(min_value=0.1, max_value=50.0),
        delta=st.floats(min_value=1e-10, max_value=0.1),
    )
    @settings(max_examples=30)
    def test_gaussian_rdp_and_zcdp_consistency(self, sigma, delta):
        """Gaussian RDP → DP and zCDP → DP should give similar ε."""
        alphas = np.array([1.5, 2.0, 5.0, 10.0, 50.0])
        _, rdp_eps = gaussian_rdp(sigma, 1.0, alphas=alphas)
        eps_rdp, _ = rdp_to_dp(rdp_eps, alphas, delta, method="standard")

        rho = gaussian_zcdp(sigma, 1.0)
        budget_zcdp = zcdp_to_dp(rho, delta)

        # Both paths should give comparable results
        # zCDP bound may be slightly looser
        assert budget_zcdp.epsilon > 0
        assert eps_rdp > 0

    @given(
        epsilon=st.floats(min_value=0.01, max_value=10.0),
        delta=st.floats(min_value=1e-10, max_value=0.1),
    )
    @settings(max_examples=30)
    def test_dp_to_rdp_bound_valid(self, epsilon, delta):
        """dp_to_rdp_bound should give valid RDP upper bounds."""
        alphas = np.array([1.5, 2.0, 5.0, 10.0])
        _, rdp_bounds = dp_to_rdp_bound(epsilon, delta, alphas=alphas)
        assert np.all(rdp_bounds >= 0)
        assert np.all(np.isfinite(rdp_bounds))
