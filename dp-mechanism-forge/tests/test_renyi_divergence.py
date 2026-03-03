"""
Tests for dp_forge.rdp.renyi_divergence — Rényi divergence computation.

Covers:
    - Discrete exact: two known distributions, verify against manual computation
    - Gaussian: verify against closed-form α/(2σ²)
    - Laplace: verify against numerical reference
    - Quadrature: verify against discrete for fine discretisation
    - Edge cases: identical distributions (divergence=0), disjoint supports
    - Vectorized computation across multiple alpha orders
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.rdp.renyi_divergence import (
    RenyiDivergenceComputer,
    RenyiDivergenceResult,
    _logsumexp,
    _validate_distribution,
)
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def computer():
    return RenyiDivergenceComputer()


@pytest.fixture
def fine_computer():
    """Computer with many quadrature points for accuracy."""
    return RenyiDivergenceComputer(quadrature_points=50000)


# =========================================================================
# Discrete exact tests
# =========================================================================


class TestExactDiscrete:
    """Tests for exact Rényi divergence on discrete distributions."""

    def test_known_distributions_alpha_2(self, computer):
        """Manual computation for α=2: D_2(P||Q) = log(Σ p²/q)."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        # D_2(P||Q) = log(sum(p^2 / q)) = log(0.25/0.4 + 0.09/0.4 + 0.04/0.2)
        # = log(0.625 + 0.225 + 0.2) = log(1.05)
        expected = math.log(0.25 / 0.4 + 0.09 / 0.4 + 0.04 / 0.2)
        result = computer.exact_discrete(p, q, alpha=2.0)
        assert result.divergence == pytest.approx(expected, rel=1e-6)
        assert result.method == "exact_discrete"

    def test_known_distributions_alpha_3(self, computer):
        """D_3(P||Q) = 1/2 * log(Σ p³/q²)."""
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        inner = (0.6 ** 3) * (0.5 ** (-2)) + (0.4 ** 3) * (0.5 ** (-2))
        expected = math.log(inner) / 2.0
        result = computer.exact_discrete(p, q, alpha=3.0)
        assert result.divergence == pytest.approx(expected, rel=1e-6)

    def test_identical_distributions_zero_divergence(self, computer):
        """D_α(P||P) = 0 for all α."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        for alpha in [1.5, 2.0, 5.0, 10.0]:
            result = computer.exact_discrete(p, p, alpha=alpha)
            assert result.divergence == pytest.approx(0.0, abs=1e-10)

    def test_kl_divergence_alpha_1(self, computer):
        """KL divergence (α→1): D_1(P||Q) = Σ p log(p/q)."""
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        expected = 0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5)
        result = computer.exact_discrete(p, q, alpha=1.0)
        assert result.divergence == pytest.approx(expected, abs=1e-6)

    def test_max_divergence_alpha_inf(self, computer):
        """Max divergence (α=∞): D_∞(P||Q) = max log(p/q)."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])
        expected = max(
            math.log(0.7 / 0.3),
            math.log(0.2 / 0.4),
            math.log(0.1 / 0.3),
        )
        result = computer.exact_discrete(p, q, alpha=float("inf"))
        assert result.divergence == pytest.approx(expected, rel=1e-6)

    def test_disjoint_support_alpha_gt_1(self, computer):
        """D_α(P||Q) = +∞ when supp(P) ⊄ supp(Q) for α > 1."""
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.0, 0.5, 0.5])
        result = computer.exact_discrete(p, q, alpha=2.0)
        assert result.divergence == float("inf")

    def test_zero_order(self, computer):
        """D_0(P||Q) = -log(Q(supp(P)))."""
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.3, 0.3, 0.4])
        expected = -math.log(0.3 + 0.3)
        result = computer.exact_discrete(p, q, alpha=0.0)
        assert result.divergence == pytest.approx(expected, rel=1e-6)

    def test_length_mismatch(self, computer):
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.3, 0.4])
        with pytest.raises(ConfigurationError):
            computer.exact_discrete(p, q, alpha=2.0)

    def test_invalid_distribution(self, computer):
        p = np.array([0.5, 0.6])  # sums to 1.1
        q = np.array([0.5, 0.5])
        with pytest.raises(ConfigurationError, match="sum"):
            computer.exact_discrete(p, q, alpha=2.0)

    def test_negative_probabilities(self, computer):
        p = np.array([0.5, -0.5, 1.0])
        q = np.array([0.3, 0.3, 0.4])
        with pytest.raises(ConfigurationError, match="negative"):
            computer.exact_discrete(p, q, alpha=2.0)

    def test_nonnegative_divergence(self, computer):
        """Rényi divergence is always non-negative."""
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        for alpha in [0.5, 1.0, 2.0, 5.0, float("inf")]:
            result = computer.exact_discrete(p, q, alpha=alpha)
            assert result.divergence >= 0


# =========================================================================
# Vectorized tests
# =========================================================================


class TestVectorized:
    """Tests for vectorized Rényi divergence computation."""

    def test_matches_scalar(self, computer):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.4, 0.4, 0.2])
        alphas = np.array([1.5, 2.0, 3.0, 5.0, 10.0])
        vectorized = computer.exact_discrete_vectorized(p, q, alphas)

        for i, alpha in enumerate(alphas):
            scalar = computer.exact_discrete(p, q, alpha=alpha)
            assert vectorized[i] == pytest.approx(scalar.divergence, rel=1e-8)

    def test_shape(self, computer):
        p = np.array([0.5, 0.5])
        q = np.array([0.4, 0.6])
        alphas = np.array([1.5, 2.0, 5.0])
        result = computer.exact_discrete_vectorized(p, q, alphas)
        assert result.shape == (3,)

    def test_monotonicity_for_different_distributions(self, computer):
        """For distributions that are not equal, divergence generally increases with α."""
        p = np.array([0.8, 0.2])
        q = np.array([0.5, 0.5])
        alphas = np.array([1.5, 2.0, 5.0, 10.0, 50.0])
        results = computer.exact_discrete_vectorized(p, q, alphas)
        # For these particular distributions with α > 1, divergence is increasing
        for i in range(len(results) - 1):
            assert results[i + 1] >= results[i] - 1e-10


# =========================================================================
# Gaussian closed-form tests
# =========================================================================


class TestGaussian:
    """Tests for closed-form Gaussian Rényi divergence."""

    def test_same_variance_formula(self, computer):
        """D_α(N(μ₁,σ²)||N(μ₂,σ²)) = α(μ₁-μ₂)²/(2σ²)."""
        mu1, mu2, sigma = 1.0, 0.0, 1.0
        alpha = 2.0
        expected = alpha * (mu1 - mu2) ** 2 / (2.0 * sigma ** 2)
        result = computer.gaussian_same_variance(mu1, mu2, sigma, alpha)
        assert result.divergence == pytest.approx(expected, rel=1e-10)

    def test_same_variance_various_alphas(self, computer):
        mu1, mu2, sigma = 0.5, 0.0, 2.0
        for alpha in [1.5, 2.0, 3.0, 10.0, 50.0]:
            expected = alpha * (mu1 - mu2) ** 2 / (2.0 * sigma ** 2)
            result = computer.gaussian_same_variance(mu1, mu2, sigma, alpha)
            assert result.divergence == pytest.approx(expected, rel=1e-10)

    def test_identical_gaussians_zero(self, computer):
        result = computer.gaussian(1.0, 1.0, 1.0, 1.0, alpha=2.0)
        assert result.divergence == pytest.approx(0.0, abs=1e-10)

    def test_general_gaussian(self, computer):
        """Test general formula with different variances."""
        mu1, sigma1 = 0.0, 1.0
        mu2, sigma2 = 1.0, 2.0
        alpha = 2.0
        result = computer.gaussian(mu1, sigma1, mu2, sigma2, alpha)
        assert result.divergence >= 0
        assert math.isfinite(result.divergence)

    def test_kl_gaussian(self, computer):
        """KL divergence: D_1(N(μ₁,σ₁²)||N(μ₂,σ₂²))."""
        mu1, sigma1 = 0.0, 1.0
        mu2, sigma2 = 1.0, 2.0
        expected = (
            math.log(sigma2 / sigma1)
            + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * sigma2 ** 2)
            - 0.5
        )
        result = computer.gaussian(mu1, sigma1, mu2, sigma2, alpha=1.0)
        assert result.divergence == pytest.approx(expected, rel=1e-6)

    def test_gaussian_invalid_sigma(self, computer):
        with pytest.raises(ConfigurationError):
            computer.gaussian(0.0, 0.0, 1.0, 1.0, alpha=2.0)

    def test_max_divergence_gaussian(self, computer):
        result = computer.gaussian(0.0, 2.0, 1.0, 3.0, alpha=float("inf"))
        assert result.divergence >= 0


# =========================================================================
# Laplace tests
# =========================================================================


class TestLaplace:
    """Tests for Laplace Rényi divergence."""

    def test_same_location_same_scale_zero(self, computer):
        result = computer.laplace(0.0, 1.0, 0.0, 1.0, alpha=2.0)
        assert result.divergence == pytest.approx(0.0, abs=1e-10)

    def test_same_scale_nonnegative(self, computer):
        result = computer.laplace(0.0, 1.0, 1.0, 1.0, alpha=2.0)
        assert result.divergence >= 0

    def test_kl_laplace(self, computer):
        """KL(Lap(μ₁,b)||Lap(μ₂,b)) reference."""
        mu1, b1 = 0.0, 1.0
        mu2, b2 = 1.0, 2.0
        expected = (
            math.log(b2 / b1)
            + abs(mu1 - mu2) / b2
            + (b1 / b2) * math.exp(-abs(mu1 - mu2) / b1)
            - 1.0
        )
        result = computer.laplace(mu1, b1, mu2, b2, alpha=1.0)
        assert result.divergence == pytest.approx(expected, rel=1e-4)

    def test_laplace_invalid_scale(self, computer):
        with pytest.raises(ConfigurationError):
            computer.laplace(0.0, 0.0, 1.0, 1.0, alpha=2.0)


# =========================================================================
# Quadrature tests
# =========================================================================


class TestQuadrature:
    """Tests for numerical quadrature Rényi divergence."""

    def test_gaussian_quadrature_vs_closed_form(self, fine_computer):
        """Quadrature should approximate closed-form for Gaussians."""
        mu1, sigma1 = 0.0, 1.0
        mu2, sigma2 = 1.0, 1.0
        alpha = 2.0

        def log_p(x):
            return -0.5 * ((x - mu1) / sigma1) ** 2 - math.log(sigma1 * math.sqrt(2 * math.pi))

        def log_q(x):
            return -0.5 * ((x - mu2) / sigma2) ** 2 - math.log(sigma2 * math.sqrt(2 * math.pi))

        quad_result = fine_computer.numerical_quadrature(log_p, log_q, alpha, lower=-20, upper=20)
        closed_result = fine_computer.gaussian_same_variance(mu1, mu2, sigma1, alpha)

        assert quad_result.divergence == pytest.approx(
            closed_result.divergence, rel=0.01
        )

    def test_quadrature_kl(self, fine_computer):
        """Quadrature KL divergence for Gaussians."""
        mu1, sigma1 = 0.0, 1.0
        mu2, sigma2 = 0.5, 1.0

        def log_p(x):
            return -0.5 * ((x - mu1) / sigma1) ** 2 - math.log(sigma1 * math.sqrt(2 * math.pi))

        def log_q(x):
            return -0.5 * ((x - mu2) / sigma2) ** 2 - math.log(sigma2 * math.sqrt(2 * math.pi))

        result = fine_computer.numerical_quadrature(log_p, log_q, alpha=1.0, lower=-20, upper=20)
        expected_kl = (mu1 - mu2) ** 2 / (2.0 * sigma2 ** 2)
        assert result.divergence == pytest.approx(expected_kl, rel=0.05)


# =========================================================================
# Compute curve and symmetrized tests
# =========================================================================


class TestComputeCurve:
    """Tests for compute_curve and symmetrized divergence."""

    def test_compute_curve(self, computer):
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        alphas = np.array([1.5, 2.0, 5.0])
        result_alphas, divergences = computer.compute_curve(p, q, alphas)
        assert len(result_alphas) == 3
        assert len(divergences) == 3
        assert np.all(divergences >= 0)

    def test_symmetrized(self, computer):
        p = np.array([0.6, 0.4])
        q = np.array([0.5, 0.5])
        sym = computer.symmetrized(p, q, alpha=2.0)
        d_pq = computer.exact_discrete(p, q, alpha=2.0).divergence
        d_qp = computer.exact_discrete(q, p, alpha=2.0).divergence
        assert sym == pytest.approx((d_pq + d_qp) / 2.0, rel=1e-10)


# =========================================================================
# Configuration tests
# =========================================================================


class TestConfiguration:
    """Tests for RenyiDivergenceComputer configuration."""

    def test_invalid_min_prob(self):
        with pytest.raises(ConfigurationError):
            RenyiDivergenceComputer(min_prob=0.0)

    def test_invalid_quadrature_points(self):
        with pytest.raises(ConfigurationError):
            RenyiDivergenceComputer(quadrature_points=50)

    def test_repr(self, computer):
        r = repr(computer)
        assert "RenyiDivergenceComputer" in r


# =========================================================================
# Property-based tests
# =========================================================================


class TestRenyiProperties:
    """Hypothesis-based property tests for Rényi divergence."""

    @given(
        alpha=st.floats(min_value=1.1, max_value=100.0),
    )
    @settings(max_examples=30)
    def test_identical_distributions_zero(self, alpha):
        """D_α(P||P) = 0 for all α."""
        computer = RenyiDivergenceComputer()
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = computer.exact_discrete(p, p, alpha=alpha)
        assert result.divergence == pytest.approx(0.0, abs=1e-8)

    @given(
        p1=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=20)
    def test_binary_nonnegative(self, p1):
        """Rényi divergence is always non-negative for binary distributions."""
        assume(0.1 < p1 < 0.9)
        computer = RenyiDivergenceComputer()
        p = np.array([p1, 1 - p1])
        q = np.array([0.5, 0.5])
        result = computer.exact_discrete(p, q, alpha=2.0)
        assert result.divergence >= -1e-10

    @given(
        mu_diff=st.floats(min_value=-5.0, max_value=5.0),
        sigma=st.floats(min_value=0.1, max_value=10.0),
        alpha=st.floats(min_value=1.5, max_value=50.0),
    )
    @settings(max_examples=30)
    def test_gaussian_same_variance_nonneg(self, mu_diff, sigma, alpha):
        """Gaussian same-variance divergence is non-negative."""
        computer = RenyiDivergenceComputer()
        result = computer.gaussian_same_variance(mu_diff, 0.0, sigma, alpha)
        assert result.divergence >= -1e-10
