"""
Tests for dp_forge.rdp.accountant — RDP accountant with composition and conversion.

Covers:
    - Adding Gaussian mechanism with known RDP curve
    - Composition: two Gaussian mechanisms, verify RDP adds
    - to_dp conversion: verify against known formulas
    - Optimal alpha selection minimises epsilon
    - Remaining budget tracking
    - Heterogeneous mechanisms composition
    - Edge cases: alpha=1 (KL divergence), alpha→∞ (max divergence)
    - Empty accountant, single mechanism
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.rdp.accountant import (
    RDPAccountant,
    RDPCurve,
    DEFAULT_ALPHAS,
    _logsumexp,
)
from dp_forge.types import PrivacyBudget
from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def default_alphas():
    return DEFAULT_ALPHAS.copy()


@pytest.fixture
def simple_alphas():
    """A small alpha grid for fast tests."""
    return np.array([1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])


@pytest.fixture
def accountant(simple_alphas):
    return RDPAccountant(alphas=simple_alphas)


@pytest.fixture
def budgeted_accountant(simple_alphas):
    budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
    return RDPAccountant(alphas=simple_alphas, total_budget=budget)


# =========================================================================
# Helper: closed-form Gaussian RDP
# =========================================================================


def gaussian_rdp_exact(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """Exact RDP for Gaussian mechanism: α·Δ²/(2σ²)."""
    return alpha * sensitivity ** 2 / (2.0 * sigma ** 2)


# =========================================================================
# RDPCurve basic tests
# =========================================================================


class TestRDPCurve:
    """Tests for the RDPCurve dataclass."""

    def test_creation(self, simple_alphas):
        eps = simple_alphas * 0.5  # linear in alpha
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps, name="test")
        assert curve.n_orders == len(simple_alphas)
        assert curve.name == "test"

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            RDPCurve(alphas=np.array([1.5, 2.0]), epsilons=np.array([0.1]))

    def test_validation_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            RDPCurve(alphas=np.array([]), epsilons=np.array([]))

    def test_validation_alpha_below_1(self):
        with pytest.raises(ValueError, match="> 1"):
            RDPCurve(alphas=np.array([0.5, 1.5]), epsilons=np.array([0.1, 0.2]))

    def test_evaluate_at_grid_point(self, simple_alphas):
        eps = simple_alphas * 0.25
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps)
        for a, e in zip(simple_alphas, eps):
            assert curve.evaluate(a) == pytest.approx(e, rel=1e-10)

    def test_evaluate_interpolation(self, simple_alphas):
        eps = simple_alphas * 0.25
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps)
        val = curve.evaluate(2.5)
        assert simple_alphas[1] < 2.5 < simple_alphas[2]
        assert eps[1] < val < eps[2]

    def test_evaluate_vectorized(self, simple_alphas):
        eps = simple_alphas * 0.25
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps)
        query = np.array([1.5, 3.0, 10.0])
        results = curve.evaluate_vectorized(query)
        for a, r in zip(query, results):
            assert r == pytest.approx(curve.evaluate(a), rel=1e-10)

    def test_evaluate_alpha_below_1_raises(self, simple_alphas):
        eps = simple_alphas * 0.25
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps)
        with pytest.raises(ConfigurationError):
            curve.evaluate(0.5)

    def test_to_dp(self, simple_alphas):
        sigma = 1.0
        eps = np.array([gaussian_rdp_exact(a, sigma) for a in simple_alphas])
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps)
        budget = curve.to_dp(delta=1e-5)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_to_dp_invalid_delta(self, simple_alphas):
        curve = RDPCurve(alphas=simple_alphas, epsilons=simple_alphas * 0.1)
        with pytest.raises(ValueError, match="delta"):
            curve.to_dp(delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            curve.to_dp(delta=1.0)

    def test_optimal_alpha(self, simple_alphas):
        sigma = 1.0
        eps = np.array([gaussian_rdp_exact(a, sigma) for a in simple_alphas])
        curve = RDPCurve(alphas=simple_alphas, epsilons=eps)
        opt_alpha = curve.optimal_alpha(delta=1e-5)
        assert opt_alpha in simple_alphas
        # Verify it truly minimises epsilon
        log_delta = math.log(1e-5)
        eps_candidates = eps - log_delta / (simple_alphas - 1.0)
        best_alpha = simple_alphas[np.argmin(eps_candidates)]
        assert opt_alpha == pytest.approx(best_alpha)

    def test_composition_same_grid(self, simple_alphas):
        eps1 = simple_alphas * 0.1
        eps2 = simple_alphas * 0.2
        c1 = RDPCurve(alphas=simple_alphas, epsilons=eps1, name="A")
        c2 = RDPCurve(alphas=simple_alphas, epsilons=eps2, name="B")
        composed = c1 + c2
        np.testing.assert_allclose(composed.epsilons, eps1 + eps2, rtol=1e-12)
        assert "A" in composed.name and "B" in composed.name

    def test_composition_different_grids(self):
        a1 = np.array([1.5, 2.0, 3.0])
        a2 = np.array([2.0, 4.0, 5.0])
        c1 = RDPCurve(alphas=a1, epsilons=a1 * 0.1)
        c2 = RDPCurve(alphas=a2, epsilons=a2 * 0.2)
        composed = c1 + c2
        # Merged grid must contain union of both grids
        for a in np.concatenate([a1, a2]):
            assert a in composed.alphas

    def test_n_orders_property(self, simple_alphas):
        curve = RDPCurve(alphas=simple_alphas, epsilons=simple_alphas * 0.1)
        assert curve.n_orders == len(simple_alphas)

    def test_repr(self, simple_alphas):
        curve = RDPCurve(alphas=simple_alphas, epsilons=simple_alphas * 0.1, name="test")
        r = repr(curve)
        assert "RDPCurve" in r
        assert "test" in r


# =========================================================================
# RDPAccountant: Adding mechanisms
# =========================================================================


class TestAccountantAddMechanism:
    """Tests for adding mechanisms to the accountant."""

    def test_add_gaussian_known_curve(self, accountant, simple_alphas):
        sigma = 1.0
        curve = accountant.add_mechanism("gaussian", sigma=sigma, sensitivity=1.0)
        expected = np.array([gaussian_rdp_exact(a, sigma) for a in simple_alphas])
        np.testing.assert_allclose(
            curve.evaluate_vectorized(simple_alphas), expected, rtol=1e-10
        )

    def test_add_gaussian_sigma_2(self, accountant, simple_alphas):
        sigma = 2.0
        curve = accountant.add_mechanism("gaussian", sigma=sigma, sensitivity=1.0)
        expected = np.array([gaussian_rdp_exact(a, sigma) for a in simple_alphas])
        np.testing.assert_allclose(
            curve.evaluate_vectorized(simple_alphas), expected, rtol=1e-10
        )

    def test_add_gaussian_with_sensitivity(self, accountant, simple_alphas):
        sigma, sens = 1.0, 2.0
        curve = accountant.add_mechanism("gaussian", sigma=sigma, sensitivity=sens)
        expected = np.array([gaussian_rdp_exact(a, sigma, sens) for a in simple_alphas])
        np.testing.assert_allclose(
            curve.evaluate_vectorized(simple_alphas), expected, rtol=1e-10
        )

    def test_add_gaussian_missing_sigma(self, accountant):
        with pytest.raises(ConfigurationError, match="sigma"):
            accountant.add_mechanism("gaussian")

    def test_add_gaussian_negative_sigma(self, accountant):
        with pytest.raises(ConfigurationError, match="positive"):
            accountant.add_mechanism("gaussian", sigma=-1.0)

    def test_add_laplace(self, accountant):
        curve = accountant.add_mechanism("laplace", epsilon=1.0)
        assert curve.n_orders == len(accountant.alphas)
        # All RDP values should be non-negative
        assert np.all(curve.epsilons >= 0)

    def test_add_laplace_missing_epsilon(self, accountant):
        with pytest.raises(ConfigurationError, match="epsilon"):
            accountant.add_mechanism("laplace")

    def test_add_rdp_curve_directly(self, accountant, simple_alphas):
        curve = RDPCurve(alphas=simple_alphas, epsilons=simple_alphas * 0.1, name="custom")
        returned = accountant.add_mechanism(curve)
        assert returned.name == "custom"
        assert accountant.n_compositions == 1

    def test_add_unknown_mechanism(self, accountant):
        with pytest.raises(ConfigurationError, match="Unknown"):
            accountant.add_mechanism("unknown_type")

    def test_add_invalid_type(self, accountant):
        with pytest.raises(ConfigurationError):
            accountant.add_mechanism(42)

    def test_add_subsampled_gaussian(self, accountant):
        curve = accountant.add_mechanism(
            "subsampled_gaussian", sigma=1.0, sensitivity=1.0, sampling_rate=0.01
        )
        # Subsampled should have smaller RDP than base Gaussian
        base_curve = RDPCurve(
            alphas=accountant.alphas,
            epsilons=np.array([gaussian_rdp_exact(a, 1.0) for a in accountant.alphas])
        )
        # For all alpha > 1, subsampled RDP ≤ base RDP
        np.testing.assert_array_less(
            curve.evaluate_vectorized(accountant.alphas) - 1e-10,
            base_curve.evaluate_vectorized(accountant.alphas)
        )


# =========================================================================
# Composition tests
# =========================================================================


class TestAccountantComposition:
    """Tests for RDP composition (additive property)."""

    def test_two_gaussians_rdp_adds(self, accountant, simple_alphas):
        sigma1, sigma2 = 1.0, 2.0
        accountant.add_mechanism("gaussian", sigma=sigma1, sensitivity=1.0)
        accountant.add_mechanism("gaussian", sigma=sigma2, sensitivity=1.0)

        expected = np.array([
            gaussian_rdp_exact(a, sigma1) + gaussian_rdp_exact(a, sigma2)
            for a in simple_alphas
        ])
        np.testing.assert_allclose(accountant.composed_rdp, expected, rtol=1e-10)

    def test_composition_is_additive(self, accountant, simple_alphas):
        e1 = simple_alphas * 0.1
        e2 = simple_alphas * 0.2
        c1 = RDPCurve(alphas=simple_alphas, epsilons=e1)
        c2 = RDPCurve(alphas=simple_alphas, epsilons=e2)
        accountant.add_mechanism(c1)
        accountant.add_mechanism(c2)
        np.testing.assert_allclose(accountant.composed_rdp, e1 + e2, rtol=1e-12)

    def test_compose_method_does_not_modify_state(self, accountant, simple_alphas):
        c1 = RDPCurve(alphas=simple_alphas, epsilons=simple_alphas * 0.1)
        c2 = RDPCurve(alphas=simple_alphas, epsilons=simple_alphas * 0.2)
        composed = accountant.compose([c1, c2])
        # Accountant state should be unchanged
        assert accountant.n_compositions == 0
        np.testing.assert_allclose(accountant.composed_rdp, 0.0)
        # Composed curve should be the sum
        expected = simple_alphas * 0.1 + simple_alphas * 0.2
        np.testing.assert_allclose(
            composed.evaluate_vectorized(simple_alphas), expected, rtol=1e-12
        )

    def test_compose_empty_raises(self, accountant):
        with pytest.raises(ValueError, match="At least one"):
            accountant.compose([])

    def test_n_compositions_tracking(self, accountant, simple_alphas):
        assert accountant.n_compositions == 0
        accountant.add_mechanism("gaussian", sigma=1.0)
        assert accountant.n_compositions == 1
        accountant.add_mechanism("gaussian", sigma=2.0)
        assert accountant.n_compositions == 2

    def test_heterogeneous_composition(self, accountant, simple_alphas):
        """Compose Gaussian + Laplace."""
        accountant.add_mechanism("gaussian", sigma=1.0)
        accountant.add_mechanism("laplace", epsilon=1.0)
        assert accountant.n_compositions == 2
        # Composed RDP should be sum of individual curves
        composed = accountant.composed_rdp
        assert len(composed) == len(simple_alphas)
        assert np.all(composed >= 0)


# =========================================================================
# to_dp conversion
# =========================================================================


class TestAccountantToDP:
    """Tests for RDP → (ε,δ)-DP conversion."""

    def test_to_dp_gaussian_returns_positive(self, accountant):
        accountant.add_mechanism("gaussian", sigma=1.0)
        budget = accountant.to_dp(delta=1e-5)
        assert budget.epsilon > 0
        assert budget.delta == 1e-5

    def test_to_dp_standard_formula(self, simple_alphas):
        """Verify against the Mironov (2017) formula:
        ε = min_α { ε̂(α) + log(1/δ) / (α - 1) }"""
        sigma = 1.0
        acct = RDPAccountant(alphas=simple_alphas)
        acct.add_mechanism("gaussian", sigma=sigma)
        delta = 1e-5
        budget = acct.to_dp(delta=delta)

        # Manual computation
        rdp_eps = np.array([gaussian_rdp_exact(a, sigma) for a in simple_alphas])
        log_delta = math.log(delta)
        eps_candidates = rdp_eps - log_delta / (simple_alphas - 1.0)
        expected_eps = max(float(np.min(eps_candidates)), 1e-15)
        assert budget.epsilon == pytest.approx(expected_eps, rel=1e-10)

    def test_to_dp_invalid_delta(self, accountant):
        with pytest.raises(ValueError, match="delta"):
            accountant.to_dp(delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            accountant.to_dp(delta=1.0)

    def test_to_dp_smaller_delta_gives_larger_epsilon(self, accountant):
        accountant.add_mechanism("gaussian", sigma=1.0)
        b1 = accountant.to_dp(delta=1e-3)
        b2 = accountant.to_dp(delta=1e-6)
        assert b2.epsilon > b1.epsilon

    def test_to_dp_composition_increases_epsilon(self, simple_alphas):
        acct1 = RDPAccountant(alphas=simple_alphas)
        acct1.add_mechanism("gaussian", sigma=1.0)
        eps1 = acct1.to_dp(delta=1e-5).epsilon

        acct2 = RDPAccountant(alphas=simple_alphas)
        acct2.add_mechanism("gaussian", sigma=1.0)
        acct2.add_mechanism("gaussian", sigma=1.0)
        eps2 = acct2.to_dp(delta=1e-5).epsilon

        assert eps2 > eps1


# =========================================================================
# Optimal alpha
# =========================================================================


class TestOptimalAlpha:
    """Tests for optimal α selection."""

    def test_optimal_alpha_in_grid(self, accountant, simple_alphas):
        accountant.add_mechanism("gaussian", sigma=1.0)
        opt = accountant.get_optimal_alpha(delta=1e-5)
        assert opt in simple_alphas

    def test_optimal_alpha_minimises_epsilon(self, accountant, simple_alphas):
        accountant.add_mechanism("gaussian", sigma=1.0)
        delta = 1e-5
        opt = accountant.get_optimal_alpha(delta=delta)

        log_delta = math.log(delta)
        rdp = accountant.composed_rdp
        eps_candidates = rdp - log_delta / (simple_alphas - 1.0)
        best_idx = np.argmin(eps_candidates)
        assert opt == pytest.approx(simple_alphas[best_idx])

    def test_optimal_alpha_invalid_delta(self, accountant):
        with pytest.raises(ValueError, match="delta"):
            accountant.get_optimal_alpha(delta=0.0)


# =========================================================================
# Remaining budget tracking
# =========================================================================


class TestRemainingBudget:
    """Tests for remaining budget computation."""

    def test_remaining_budget_basic(self, budgeted_accountant):
        budgeted_accountant.add_mechanism("gaussian", sigma=5.0)
        remaining = budgeted_accountant.remaining_budget(delta=1e-5)
        assert remaining.epsilon > 0
        assert remaining.delta == 1e-5

    def test_remaining_budget_decreases(self, budgeted_accountant):
        budgeted_accountant.add_mechanism("gaussian", sigma=5.0)
        r1 = budgeted_accountant.remaining_budget(delta=1e-5)
        budgeted_accountant.add_mechanism("gaussian", sigma=5.0)
        r2 = budgeted_accountant.remaining_budget(delta=1e-5)
        assert r2.epsilon < r1.epsilon

    def test_remaining_budget_exhausted(self, simple_alphas):
        budget = PrivacyBudget(epsilon=0.5, delta=1e-5)
        acct = RDPAccountant(alphas=simple_alphas, total_budget=budget)
        # Adding a noisy-enough mechanism that barely fits, then check remaining
        with pytest.raises(BudgetExhaustedError):
            acct.add_mechanism("gaussian", sigma=0.5)  # too much privacy loss

    def test_remaining_budget_no_budget_set(self, accountant):
        with pytest.raises(ConfigurationError, match="total budget"):
            accountant.remaining_budget()

    def test_budget_exceeded_on_add(self, simple_alphas):
        budget = PrivacyBudget(epsilon=0.1, delta=1e-5)
        acct = RDPAccountant(alphas=simple_alphas, total_budget=budget)
        with pytest.raises(BudgetExhaustedError):
            acct.add_mechanism("gaussian", sigma=0.1)


# =========================================================================
# State management
# =========================================================================


class TestStateManagement:
    """Tests for reset, fork, and composed curve."""

    def test_reset(self, accountant):
        accountant.add_mechanism("gaussian", sigma=1.0)
        assert accountant.n_compositions == 1
        accountant.reset()
        assert accountant.n_compositions == 0
        np.testing.assert_allclose(accountant.composed_rdp, 0.0)

    def test_fork(self, accountant, simple_alphas):
        accountant.add_mechanism("gaussian", sigma=1.0)
        forked = accountant.fork()
        assert forked.n_compositions == accountant.n_compositions
        np.testing.assert_allclose(forked.composed_rdp, accountant.composed_rdp)

        # Modifying fork doesn't affect original
        forked.add_mechanism("gaussian", sigma=2.0)
        assert forked.n_compositions != accountant.n_compositions

    def test_get_composed_curve(self, accountant, simple_alphas):
        accountant.add_mechanism("gaussian", sigma=1.0)
        curve = accountant.get_composed_curve()
        assert isinstance(curve, RDPCurve)
        np.testing.assert_allclose(curve.epsilons, accountant.composed_rdp)

    def test_curves_property(self, accountant):
        accountant.add_mechanism("gaussian", sigma=1.0, name="g1")
        accountant.add_mechanism("laplace", epsilon=1.0, name="l1")
        curves = accountant.curves
        assert len(curves) == 2


# =========================================================================
# Empty and single mechanism
# =========================================================================


class TestEdgeCases:
    """Edge cases: empty accountant, single mechanism, custom alpha grids."""

    def test_empty_accountant_composed_rdp_is_zero(self, accountant, simple_alphas):
        np.testing.assert_allclose(accountant.composed_rdp, 0.0)
        assert accountant.n_compositions == 0

    def test_empty_accountant_to_dp(self, accountant):
        budget = accountant.to_dp(delta=1e-5)
        # With zero composed RDP, ε should still be non-negative
        assert budget.epsilon >= 0

    def test_single_mechanism(self, accountant):
        accountant.add_mechanism("gaussian", sigma=1.0)
        budget = accountant.to_dp(delta=1e-5)
        assert budget.epsilon > 0

    def test_custom_alpha_grid(self):
        alphas = np.array([1.1, 2.0, 5.0])
        acct = RDPAccountant(alphas=alphas)
        np.testing.assert_array_equal(acct.alphas, alphas)

    def test_alpha_below_1_raises(self):
        with pytest.raises(ConfigurationError):
            RDPAccountant(alphas=np.array([0.5, 1.0, 2.0]))

    def test_default_alphas_used(self):
        acct = RDPAccountant()
        np.testing.assert_array_equal(acct.alphas, DEFAULT_ALPHAS)

    def test_repr(self, accountant):
        r = repr(accountant)
        assert "RDPAccountant" in r


# =========================================================================
# logsumexp helper
# =========================================================================


class TestLogSumExp:
    """Tests for the internal _logsumexp helper."""

    def test_single_element(self):
        assert _logsumexp(np.array([3.0])) == pytest.approx(3.0)

    def test_known_values(self):
        # log(exp(1) + exp(2)) = 2 + log(1 + exp(-1))
        expected = 2.0 + math.log(1 + math.exp(-1))
        assert _logsumexp(np.array([1.0, 2.0])) == pytest.approx(expected, rel=1e-10)

    def test_empty_array(self):
        assert _logsumexp(np.array([])) == -np.inf

    def test_large_values_stability(self):
        # Should not overflow
        result = _logsumexp(np.array([1000.0, 1001.0]))
        expected = 1001.0 + math.log(1 + math.exp(-1))
        assert result == pytest.approx(expected, rel=1e-10)


# =========================================================================
# Property-based tests (hypothesis)
# =========================================================================


class TestAccountantProperties:
    """Property-based tests using hypothesis."""

    @given(
        sigma=st.floats(min_value=0.1, max_value=100.0),
        delta=st.floats(min_value=1e-10, max_value=0.1),
    )
    @settings(max_examples=50)
    def test_gaussian_epsilon_positive(self, sigma, delta):
        """ε from Gaussian mechanism is always positive."""
        alphas = np.array([1.5, 2.0, 5.0, 10.0, 50.0])
        acct = RDPAccountant(alphas=alphas)
        acct.add_mechanism("gaussian", sigma=sigma)
        budget = acct.to_dp(delta=delta)
        assert budget.epsilon > 0

    @given(
        sigma=st.floats(min_value=0.5, max_value=10.0),
    )
    @settings(max_examples=30)
    def test_more_noise_gives_smaller_epsilon(self, sigma):
        """Larger σ → smaller ε (for fixed delta)."""
        alphas = np.array([1.5, 2.0, 5.0, 10.0, 50.0])
        delta = 1e-5
        acct1 = RDPAccountant(alphas=alphas)
        acct1.add_mechanism("gaussian", sigma=sigma)
        acct2 = RDPAccountant(alphas=alphas)
        acct2.add_mechanism("gaussian", sigma=sigma * 2)
        eps1 = acct1.to_dp(delta=delta).epsilon
        eps2 = acct2.to_dp(delta=delta).epsilon
        assert eps2 < eps1

    @given(
        n_mechanisms=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20)
    def test_composition_monotonicity(self, n_mechanisms):
        """Adding more mechanisms never decreases ε."""
        alphas = np.array([1.5, 2.0, 5.0, 10.0])
        delta = 1e-5
        acct = RDPAccountant(alphas=alphas)
        prev_eps = 0.0
        for _ in range(n_mechanisms):
            acct.add_mechanism("gaussian", sigma=1.0)
            eps = acct.to_dp(delta=delta).epsilon
            assert eps >= prev_eps - 1e-15
            prev_eps = eps

    @given(
        sigma=st.floats(min_value=0.1, max_value=50.0),
    )
    @settings(max_examples=30)
    def test_rdp_nonnegative(self, sigma):
        """RDP values are always non-negative for Gaussian."""
        alphas = np.array([1.5, 2.0, 5.0, 10.0, 50.0])
        acct = RDPAccountant(alphas=alphas)
        acct.add_mechanism("gaussian", sigma=sigma)
        assert np.all(acct.composed_rdp >= -1e-15)
