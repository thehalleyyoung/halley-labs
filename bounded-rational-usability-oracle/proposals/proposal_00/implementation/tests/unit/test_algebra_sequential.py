"""
Unit tests for usability_oracle.algebra.sequential — Sequential composition ⊕.

Tests the SequentialComposer class which implements sequential task composition
with coupling-based correlation between successive cognitive cost elements.

Mathematical model under test:
    μ_{a⊕b}  = μ_a + μ_b + ρ·√(σ²_a·σ²_b)
    σ²_{a⊕b} = σ²_a + σ²_b + 2ρ·√(σ²_a·σ²_b)
    κ_{a⊕b}  = (κ_a·(σ²_a)^{3/2} + κ_b·(σ²_b)^{3/2}) / (σ²_{a⊕b})^{3/2}
    λ_{a⊕b}  = max(λ_a, λ_b) + ρ·min(λ_a, λ_b)
"""

import math
import pytest
import numpy as np

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.sequential import SequentialComposer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def composer():
    """Return a fresh SequentialComposer instance."""
    return SequentialComposer()


@pytest.fixture
def elem_a():
    """A typical cost element for a visual search task."""
    return CostElement(mu=2.0, sigma_sq=0.5, kappa=0.1, lambda_=0.05)


@pytest.fixture
def elem_b():
    """A typical cost element for a click/response task."""
    return CostElement(mu=1.0, sigma_sq=0.2, kappa=0.05, lambda_=0.02)


@pytest.fixture
def elem_c():
    """A third cost element for chain tests."""
    return CostElement(mu=0.5, sigma_sq=0.1, kappa=0.0, lambda_=0.01)


@pytest.fixture
def zero():
    """The additive identity element."""
    return CostElement.zero()


# ---------------------------------------------------------------------------
# Basic composition tests
# ---------------------------------------------------------------------------

class TestComposeBasic:
    """Tests for basic sequential composition with zero coupling."""

    def test_compose_mu_adds(self, composer, elem_a, elem_b):
        """With zero coupling, composed μ equals sum of individual μ values."""
        result = composer.compose(elem_a, elem_b, coupling=0.0)
        assert math.isclose(result.mu, elem_a.mu + elem_b.mu, rel_tol=1e-10)

    def test_compose_sigma_sq_adds(self, composer, elem_a, elem_b):
        """With zero coupling, composed σ² equals sum of individual σ² values."""
        result = composer.compose(elem_a, elem_b, coupling=0.0)
        assert math.isclose(result.sigma_sq, elem_a.sigma_sq + elem_b.sigma_sq, rel_tol=1e-10)

    def test_compose_kappa_weighted(self, composer, elem_a, elem_b):
        """Composed kappa is a variance-weighted combination of individual kappas."""
        result = composer.compose(elem_a, elem_b, coupling=0.0)
        total_var = elem_a.sigma_sq + elem_b.sigma_sq
        expected_kappa = (
            elem_a.kappa * elem_a.sigma_sq ** 1.5
            + elem_b.kappa * elem_b.sigma_sq ** 1.5
        ) / total_var ** 1.5
        assert math.isclose(result.kappa, expected_kappa, rel_tol=1e-8)

    def test_compose_lambda_max(self, composer, elem_a, elem_b):
        """With zero coupling, composed λ equals max of individual λ values."""
        result = composer.compose(elem_a, elem_b, coupling=0.0)
        assert math.isclose(result.lambda_, max(elem_a.lambda_, elem_b.lambda_), rel_tol=1e-10)

    def test_compose_returns_cost_element(self, composer, elem_a, elem_b):
        """compose() returns a CostElement instance."""
        result = composer.compose(elem_a, elem_b)
        assert isinstance(result, CostElement)

    def test_compose_result_is_valid(self, composer, elem_a, elem_b):
        """The composed result satisfies validity constraints."""
        result = composer.compose(elem_a, elem_b, coupling=0.3)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Coupling tests
# ---------------------------------------------------------------------------

class TestComposeCoupling:
    """Tests for sequential composition with non-zero coupling."""

    def test_coupling_increases_mu(self, composer, elem_a, elem_b):
        """Positive coupling increases the composed mean cost."""
        uncoupled = composer.compose(elem_a, elem_b, coupling=0.0)
        coupled = composer.compose(elem_a, elem_b, coupling=0.5)
        assert coupled.mu > uncoupled.mu

    def test_coupling_increases_sigma_sq(self, composer, elem_a, elem_b):
        """Positive coupling increases the composed variance."""
        uncoupled = composer.compose(elem_a, elem_b, coupling=0.0)
        coupled = composer.compose(elem_a, elem_b, coupling=0.5)
        assert coupled.sigma_sq > uncoupled.sigma_sq

    def test_coupling_mu_formula(self, composer, elem_a, elem_b):
        """Verify the exact coupling formula for μ: μ_a + μ_b + ρ·√(σ²_a·σ²_b)."""
        rho = 0.4
        result = composer.compose(elem_a, elem_b, coupling=rho)
        sqrt_cross = math.sqrt(elem_a.sigma_sq * elem_b.sigma_sq)
        expected_mu = elem_a.mu + elem_b.mu + rho * sqrt_cross
        assert math.isclose(result.mu, expected_mu, rel_tol=1e-10)

    def test_coupling_sigma_sq_formula(self, composer, elem_a, elem_b):
        """Verify the exact coupling formula for σ²: σ²_a + σ²_b + 2ρ·√(σ²_a·σ²_b)."""
        rho = 0.4
        result = composer.compose(elem_a, elem_b, coupling=rho)
        sqrt_cross = math.sqrt(elem_a.sigma_sq * elem_b.sigma_sq)
        expected_var = elem_a.sigma_sq + elem_b.sigma_sq + 2.0 * rho * sqrt_cross
        assert math.isclose(result.sigma_sq, expected_var, rel_tol=1e-10)

    def test_coupling_lambda_formula(self, composer, elem_a, elem_b):
        """Verify λ formula: max(λ_a, λ_b) + ρ·min(λ_a, λ_b)."""
        rho = 0.6
        result = composer.compose(elem_a, elem_b, coupling=rho)
        expected_lambda = max(elem_a.lambda_, elem_b.lambda_) + rho * min(elem_a.lambda_, elem_b.lambda_)
        assert math.isclose(result.lambda_, expected_lambda, rel_tol=1e-10)

    def test_full_coupling_maximises(self, composer, elem_a, elem_b):
        """Full coupling (ρ=1) gives the maximum composed cost."""
        partial = composer.compose(elem_a, elem_b, coupling=0.5)
        full = composer.compose(elem_a, elem_b, coupling=1.0)
        assert full.mu >= partial.mu
        assert full.sigma_sq >= partial.sigma_sq

    def test_coupling_monotonic_in_rho(self, composer, elem_a, elem_b):
        """Composed μ is monotonically non-decreasing in ρ."""
        mus = [composer.compose(elem_a, elem_b, coupling=r).mu for r in np.linspace(0, 1, 20)]
        for i in range(1, len(mus)):
            assert mus[i] >= mus[i - 1] - 1e-12


# ---------------------------------------------------------------------------
# Identity and zero element
# ---------------------------------------------------------------------------

class TestComposeIdentity:
    """Tests for the identity property: a ⊕ 0 = a."""

    def test_compose_with_zero_right(self, composer, elem_a, zero):
        """Composing with zero on the right returns the original element."""
        result = composer.compose(elem_a, zero, coupling=0.0)
        assert math.isclose(result.mu, elem_a.mu, abs_tol=1e-12)
        assert math.isclose(result.sigma_sq, elem_a.sigma_sq, abs_tol=1e-12)
        assert math.isclose(result.kappa, elem_a.kappa, abs_tol=1e-12)
        assert math.isclose(result.lambda_, elem_a.lambda_, abs_tol=1e-12)

    def test_compose_with_zero_left(self, composer, elem_a, zero):
        """Composing with zero on the left returns the original element."""
        result = composer.compose(zero, elem_a, coupling=0.0)
        assert math.isclose(result.mu, elem_a.mu, abs_tol=1e-12)
        assert math.isclose(result.sigma_sq, elem_a.sigma_sq, abs_tol=1e-12)

    def test_compose_zero_with_zero(self, composer, zero):
        """Composing zero with zero yields zero."""
        result = composer.compose(zero, zero, coupling=0.0)
        assert result == CostElement.zero()


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------

class TestComposeChain:
    """Tests for compose_chain — composing a list of elements sequentially."""

    def test_chain_two_equals_compose(self, composer, elem_a, elem_b):
        """Chaining two elements equals a single compose call."""
        chain = composer.compose_chain([elem_a, elem_b])
        direct = composer.compose(elem_a, elem_b, coupling=0.0)
        assert math.isclose(chain.mu, direct.mu, rel_tol=1e-10)
        assert math.isclose(chain.sigma_sq, direct.sigma_sq, rel_tol=1e-10)

    def test_chain_three_elements(self, composer, elem_a, elem_b, elem_c):
        """Chaining three elements produces a valid result."""
        result = composer.compose_chain([elem_a, elem_b, elem_c])
        assert result.mu >= elem_a.mu + elem_b.mu + elem_c.mu - 1e-10
        assert result.is_valid

    def test_chain_with_couplings(self, composer, elem_a, elem_b, elem_c):
        """Chaining with explicit coupling values."""
        result = composer.compose_chain([elem_a, elem_b, elem_c], couplings=[0.2, 0.3])
        uncoupled = composer.compose_chain([elem_a, elem_b, elem_c])
        assert result.mu > uncoupled.mu

    def test_chain_empty_returns_zero(self, composer):
        """Chaining an empty list returns the zero element."""
        result = composer.compose_chain([])
        assert result == CostElement.zero()

    def test_chain_single_returns_element(self, composer, elem_a):
        """Chaining a single element returns that element."""
        result = composer.compose_chain([elem_a])
        assert result.mu == elem_a.mu

    def test_chain_wrong_couplings_length_raises(self, composer, elem_a, elem_b):
        """Mismatched coupling list length raises ValueError."""
        with pytest.raises(ValueError, match="coupling"):
            composer.compose_chain([elem_a, elem_b], couplings=[0.1, 0.2])


# ---------------------------------------------------------------------------
# Interval composition
# ---------------------------------------------------------------------------

class TestComposeInterval:
    """Tests for compose_interval — coupling as an interval."""

    def test_interval_returns_two_bounds(self, composer, elem_a, elem_b):
        """compose_interval returns a (lower, upper) tuple."""
        lo, hi = composer.compose_interval(elem_a, elem_b, (0.0, 0.5))
        assert isinstance(lo, CostElement)
        assert isinstance(hi, CostElement)

    def test_interval_lower_leq_upper(self, composer, elem_a, elem_b):
        """Lower bound μ ≤ upper bound μ."""
        lo, hi = composer.compose_interval(elem_a, elem_b, (0.1, 0.8))
        assert lo.mu <= hi.mu + 1e-12

    def test_interval_matches_compose(self, composer, elem_a, elem_b):
        """Bounds match individual compose calls at interval endpoints."""
        lo, hi = composer.compose_interval(elem_a, elem_b, (0.2, 0.7))
        direct_lo = composer.compose(elem_a, elem_b, coupling=0.2)
        direct_hi = composer.compose(elem_a, elem_b, coupling=0.7)
        assert math.isclose(lo.mu, direct_lo.mu, rel_tol=1e-10)
        assert math.isclose(hi.mu, direct_hi.mu, rel_tol=1e-10)

    def test_interval_swapped_bounds_auto_fix(self, composer, elem_a, elem_b):
        """Swapped interval bounds are auto-corrected."""
        lo, hi = composer.compose_interval(elem_a, elem_b, (0.8, 0.2))
        assert lo.mu <= hi.mu + 1e-12


# ---------------------------------------------------------------------------
# Matrix composition
# ---------------------------------------------------------------------------

class TestComposeMatrix:
    """Tests for compose_matrix — n-ary composition with pairwise couplings."""

    def test_matrix_two_elements(self, composer, elem_a, elem_b):
        """Matrix composition with 2 elements matches direct compose."""
        mat = np.array([[0.0, 0.3], [0.3, 0.0]])
        result = composer.compose_matrix([elem_a, elem_b], mat)
        direct = composer.compose(elem_a, elem_b, coupling=0.3)
        assert math.isclose(result.mu, direct.mu, rel_tol=1e-10)

    def test_matrix_wrong_shape_raises(self, composer, elem_a, elem_b):
        """Mismatched matrix shape raises ValueError."""
        mat = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
        with pytest.raises(ValueError, match="shape"):
            composer.compose_matrix([elem_a, elem_b], mat)

    def test_matrix_empty(self, composer):
        """Matrix composition with empty list returns zero."""
        mat = np.empty((0, 0))
        result = composer.compose_matrix([], mat)
        assert result == CostElement.zero()


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class TestSensitivity:
    """Tests for sensitivity — partial derivatives of composed cost."""

    def test_sensitivity_returns_dict(self, composer, elem_a, elem_b):
        """sensitivity() returns a dict with expected keys."""
        sens = composer.sensitivity(elem_a, elem_b, coupling=0.2)
        assert isinstance(sens, dict)
        expected_keys = {"d_mu_a", "d_mu_b", "d_sigma_sq_a", "d_sigma_sq_b", "d_coupling"}
        assert expected_keys == set(sens.keys())

    def test_sensitivity_d_mu_a_approx_one(self, composer, elem_a, elem_b):
        """∂μ_composed/∂μ_a ≈ 1 (mean adds linearly when coupling is 0)."""
        sens = composer.sensitivity(elem_a, elem_b, coupling=0.0)
        assert math.isclose(sens["d_mu_a"].mu, 1.0, abs_tol=1e-4)

    def test_sensitivity_d_mu_b_approx_one(self, composer, elem_a, elem_b):
        """∂μ_composed/∂μ_b ≈ 1 (mean adds linearly when coupling is 0)."""
        sens = composer.sensitivity(elem_a, elem_b, coupling=0.0)
        assert math.isclose(sens["d_mu_b"].mu, 1.0, abs_tol=1e-4)

    def test_sensitivity_d_coupling_nonneg(self, composer, elem_a, elem_b):
        """∂μ/∂ρ ≈ √(σ²_a·σ²_b) ≥ 0."""
        sens = composer.sensitivity(elem_a, elem_b, coupling=0.3)
        expected = math.sqrt(elem_a.sigma_sq * elem_b.sigma_sq)
        assert math.isclose(sens["d_coupling"].mu, expected, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for input validation and error handling."""

    def test_negative_coupling_raises(self, composer, elem_a, elem_b):
        """Coupling < 0 raises ValueError."""
        with pytest.raises(ValueError, match="[Cc]oupling"):
            composer.compose(elem_a, elem_b, coupling=-0.1)

    def test_coupling_above_one_raises(self, composer, elem_a, elem_b):
        """Coupling > 1 raises ValueError."""
        with pytest.raises(ValueError, match="[Cc]oupling"):
            composer.compose(elem_a, elem_b, coupling=1.5)

    def test_coupling_boundary_zero_ok(self, composer, elem_a, elem_b):
        """Coupling exactly 0 is valid."""
        result = composer.compose(elem_a, elem_b, coupling=0.0)
        assert result.is_valid

    def test_coupling_boundary_one_ok(self, composer, elem_a, elem_b):
        """Coupling exactly 1 is valid."""
        result = composer.compose(elem_a, elem_b, coupling=1.0)
        assert result.is_valid

    def test_degenerate_elements(self, composer):
        """Composing two degenerate (zero variance) elements works correctly."""
        a = CostElement(mu=1.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)
        b = CostElement(mu=2.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)
        result = composer.compose(a, b, coupling=0.5)
        assert math.isclose(result.mu, 3.0, rel_tol=1e-10)
        assert result.sigma_sq < 1e-12


# ---------------------------------------------------------------------------
# Soundness properties
# ---------------------------------------------------------------------------

class TestSoundnessProperties:
    """Tests verifying algebraic soundness properties of sequential composition."""

    def test_monotonicity(self, composer, elem_a, elem_b):
        """Composed μ ≥ max(μ_a, μ_b) for any valid coupling."""
        for rho in [0.0, 0.3, 0.7, 1.0]:
            result = composer.compose(elem_a, elem_b, coupling=rho)
            assert result.mu >= max(elem_a.mu, elem_b.mu) - 1e-12

    def test_variance_lower_bound(self, composer, elem_a, elem_b):
        """Composed σ² ≥ max(σ²_a, σ²_b) for any valid coupling."""
        for rho in [0.0, 0.5, 1.0]:
            result = composer.compose(elem_a, elem_b, coupling=rho)
            assert result.sigma_sq >= max(elem_a.sigma_sq, elem_b.sigma_sq) - 1e-12

    def test_lambda_clamped_to_one(self, composer):
        """Composed λ is clamped to [0, 1] even with high coupling and input λ."""
        a = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.8)
        b = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.9)
        result = composer.compose(a, b, coupling=1.0)
        assert result.lambda_ <= 1.0
