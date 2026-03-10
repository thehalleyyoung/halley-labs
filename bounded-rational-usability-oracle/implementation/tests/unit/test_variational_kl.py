"""Tests for usability_oracle.variational.kl_divergence.

Verifies KL divergence, mutual information, Jensen–Shannon divergence,
and Rényi divergence against known mathematical properties and analytical
closed-form values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.variational.kl_divergence import (
    compute_kl_divergence,
    compute_kl_discrete,
    compute_kl_gaussian,
    compute_mutual_information,
    compute_policy_kl,
    renyi_divergence,
    symmetric_kl,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def uniform3() -> np.ndarray:
    """Uniform distribution over 3 elements."""
    return np.array([1 / 3, 1 / 3, 1 / 3])


@pytest.fixture
def uniform4() -> np.ndarray:
    """Uniform distribution over 4 elements."""
    return np.ones(4) / 4


@pytest.fixture
def peaked3() -> np.ndarray:
    """Peaked distribution over 3 elements."""
    return np.array([0.8, 0.1, 0.1])


@pytest.fixture
def independent_joint() -> np.ndarray:
    """Joint distribution for two independent variables."""
    p_x = np.array([0.5, 0.5])
    p_y = np.array([0.3, 0.7])
    return np.outer(p_x, p_y)


@pytest.fixture
def correlated_joint() -> np.ndarray:
    """Joint distribution with positive correlation."""
    return np.array([[0.4, 0.1], [0.1, 0.4]])


# =====================================================================
# KL divergence — basic properties
# =====================================================================

class TestKLDivergenceBasicProperties:
    """Test fundamental mathematical properties of KL divergence."""

    def test_identical_distributions_zero(self, uniform3: np.ndarray) -> None:
        """D_KL(p || p) = 0 for any valid distribution p."""
        assert compute_kl_divergence(uniform3, uniform3) == pytest.approx(0.0, abs=1e-12)

    def test_identical_peaked_zero(self, peaked3: np.ndarray) -> None:
        """D_KL(p || p) = 0 even for non-uniform distributions."""
        assert compute_kl_divergence(peaked3, peaked3) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("p,q", [
        (np.array([0.7, 0.3]), np.array([0.5, 0.5])),
        (np.array([0.9, 0.05, 0.05]), np.array([1 / 3, 1 / 3, 1 / 3])),
        (np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.6, 0.3])),
    ])
    def test_non_negative_gibbs_inequality(self, p: np.ndarray, q: np.ndarray) -> None:
        """Gibbs' inequality: D_KL(p || q) ≥ 0 for all p, q."""
        kl = compute_kl_divergence(p, q)
        assert kl >= -1e-12, f"KL divergence should be non-negative, got {kl}"

    def test_asymmetry(self) -> None:
        """KL(P || Q) ≠ KL(Q || P) in general (asymmetry)."""
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp, abs=1e-6), (
            "KL divergence should be asymmetric"
        )


# =====================================================================
# KL divergence — analytical values
# =====================================================================

class TestKLDivergenceAnalytical:
    """Test KL against analytically known values."""

    @pytest.mark.parametrize("n,expected_kl", [
        (3, 0.0),
        (5, 0.0),
        (10, 0.0),
    ])
    def test_uniform_same_size_zero_kl(
        self, n: int, expected_kl: float
    ) -> None:
        """D_KL(Uniform(n) || Uniform(n)) = 0."""
        p = np.ones(n) / n
        q = np.ones(n) / n
        kl = compute_kl_divergence(p, q)
        assert kl == pytest.approx(expected_kl, abs=1e-10)

    def test_uniform_p_subset_of_q_gives_log_ratio(self) -> None:
        """D_KL(Uniform(n) padded || Uniform(m)) = log(m/n) when n < m.

        p is uniform on first n of m elements (with zeros elsewhere),
        q is uniform on all m elements.  KL = log(m/n).
        """
        n, m = 2, 4
        p = np.zeros(m)
        p[:n] = 1.0 / n
        q = np.ones(m) / m
        kl = compute_kl_divergence(p, q)
        expected = math.log(m / n)  # = log(2)
        assert kl == pytest.approx(expected, abs=1e-6)

    def test_kl_binary_distributions(self) -> None:
        """Known analytical KL for Bernoulli: D_KL(p || q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))."""
        p_val = 0.8
        q_val = 0.5
        expected = p_val * math.log(p_val / q_val) + (1 - p_val) * math.log(
            (1 - p_val) / (1 - q_val)
        )
        p = np.array([p_val, 1 - p_val])
        q = np.array([q_val, 1 - q_val])
        kl = compute_kl_divergence(p, q)
        assert kl == pytest.approx(expected, rel=1e-8)


# =====================================================================
# KL for Gaussians
# =====================================================================

class TestKLGaussian:
    """Test closed-form KL for univariate Gaussians."""

    def test_identical_gaussians(self) -> None:
        """KL between identical Gaussians is 0."""
        kl = compute_kl_gaussian(0.0, 1.0, 0.0, 1.0)
        assert kl == pytest.approx(0.0, abs=1e-12)

    def test_different_means(self) -> None:
        """KL between Gaussians differing only in mean."""
        mu1, sigma = 0.0, 1.0
        mu2 = 1.0
        expected = (sigma ** 2 + (mu1 - mu2) ** 2) / (2 * sigma ** 2) - 0.5
        kl = compute_kl_gaussian(mu1, sigma, mu2, sigma)
        assert kl == pytest.approx(expected, rel=1e-8)

    def test_different_variances(self) -> None:
        """KL between Gaussians differing only in variance."""
        mu = 0.0
        sigma1, sigma2 = 1.0, 2.0
        expected = math.log(sigma2 / sigma1) + sigma1 ** 2 / (2 * sigma2 ** 2) - 0.5
        kl = compute_kl_gaussian(mu, sigma1, mu, sigma2)
        assert kl == pytest.approx(expected, rel=1e-8)

    def test_non_negative(self) -> None:
        """Gaussian KL should always be non-negative."""
        kl = compute_kl_gaussian(3.0, 2.0, -1.0, 0.5)
        assert kl >= 0.0

    def test_invalid_sigma_raises(self) -> None:
        """sigma ≤ 0 should raise ValueError."""
        with pytest.raises(ValueError, match="sigma1"):
            compute_kl_gaussian(0.0, 0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="sigma2"):
            compute_kl_gaussian(0.0, 1.0, 0.0, -1.0)

    def test_gaussian_kl_against_numerical(self) -> None:
        """Compare closed-form Gaussian KL against numerical integration."""
        mu1, sigma1 = 1.0, 0.5
        mu2, sigma2 = 0.0, 1.0
        # Numerical: discretize and compute discrete KL
        x = np.linspace(-6, 8, 10000)
        p = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
        q = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
        dx = x[1] - x[0]
        p = p * dx  # approximate probabilities
        q = q * dx
        mask = p > 1e-15
        numerical_kl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
        closed_form_kl = compute_kl_gaussian(mu1, sigma1, mu2, sigma2)
        assert closed_form_kl == pytest.approx(numerical_kl, rel=0.01)


# =====================================================================
# Mutual information
# =====================================================================

class TestMutualInformation:
    """Test mutual information I(X; Y) computation."""

    def test_independent_variables_zero(self, independent_joint: np.ndarray) -> None:
        """I(X; Y) = 0 when X and Y are independent."""
        mi = compute_mutual_information(independent_joint)
        assert mi == pytest.approx(0.0, abs=1e-10)

    def test_correlated_variables_positive(self, correlated_joint: np.ndarray) -> None:
        """I(X; Y) > 0 when X and Y are correlated."""
        mi = compute_mutual_information(correlated_joint)
        assert mi > 0.0

    def test_perfect_correlation(self) -> None:
        """I(X; Y) = H(X) for perfectly correlated variables."""
        # Diagonal joint → perfect correlation
        joint = np.diag([0.25, 0.25, 0.25, 0.25])
        mi = compute_mutual_information(joint)
        expected_hx = math.log(4)  # H(X) = log(4) for uniform
        assert mi == pytest.approx(expected_hx, rel=1e-6)

    def test_invalid_joint_shape_raises(self) -> None:
        """1-D array should raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            compute_mutual_information(np.array([0.5, 0.5]))


# =====================================================================
# Jensen–Shannon divergence
# =====================================================================

class TestJensenShannonDivergence:
    """Test JSD properties."""

    def test_symmetric(self) -> None:
        """JSD(p || q) = JSD(q || p)."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])
        assert symmetric_kl(p, q) == pytest.approx(symmetric_kl(q, p), abs=1e-12)

    @pytest.mark.parametrize("p,q", [
        (np.array([0.5, 0.5]), np.array([0.9, 0.1])),
        (np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.8, 0.1, 0.1])),
        (np.array([0.1, 0.9]), np.array([0.9, 0.1])),
    ])
    def test_bounded_by_log2(self, p: np.ndarray, q: np.ndarray) -> None:
        """JSD is bounded above by ln(2)."""
        jsd = symmetric_kl(p, q)
        assert jsd <= math.log(2) + 1e-10

    def test_identical_distributions_zero(self) -> None:
        """JSD(p || p) = 0."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        assert symmetric_kl(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_non_negative(self) -> None:
        """JSD is always non-negative."""
        p = np.array([0.6, 0.4])
        q = np.array([0.3, 0.7])
        assert symmetric_kl(p, q) >= -1e-12


# =====================================================================
# Rényi divergence
# =====================================================================

class TestRenyiDivergence:
    """Test Rényi divergence D_α(p || q)."""

    def test_reduces_to_kl_at_alpha_1(self) -> None:
        """D_α(p || q) → D_KL(p || q) as α → 1."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])
        renyi_val = renyi_divergence(p, q, alpha=1.0 + 1e-12)
        kl_val = compute_kl_divergence(p, q)
        assert renyi_val == pytest.approx(kl_val, rel=0.01)

    def test_non_negative(self) -> None:
        """Rényi divergence is non-negative for all α > 0."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.4, 0.5])
        for alpha in [0.5, 1.5, 2.0, 5.0]:
            assert renyi_divergence(p, q, alpha) >= -1e-12

    def test_identical_distributions_zero(self) -> None:
        """D_α(p || p) = 0 for any α."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        for alpha in [0.5, 1.5, 2.0]:
            assert renyi_divergence(p, p, alpha) == pytest.approx(0.0, abs=1e-10)

    def test_invalid_alpha_raises(self) -> None:
        """α ≤ 0 should raise ValueError."""
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="alpha"):
            renyi_divergence(p, q, alpha=-1.0)

    @pytest.mark.parametrize("alpha", [0.5, 2.0, 3.0])
    def test_monotonic_in_alpha(self, alpha: float) -> None:
        """For fixed p, q, D_α is non-decreasing in α (for α ≥ 0)."""
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([1 / 3, 1 / 3, 1 / 3])
        d_low = renyi_divergence(p, q, 0.5)
        d_high = renyi_divergence(p, q, alpha)
        if alpha >= 0.5:
            assert d_high >= d_low - 1e-10


# =====================================================================
# Zero probability handling
# =====================================================================

class TestZeroProbabilityHandling:
    """Test edge cases with zero-probability entries."""

    def test_zero_in_q_where_p_positive_gives_inf(self) -> None:
        """D_KL = +inf when p(x) > 0 but q(x) = 0."""
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        kl = compute_kl_divergence(p, q)
        assert kl == float("inf")

    def test_zero_in_p_handled_gracefully(self) -> None:
        """Zero entries in p should contribute 0 (by convention 0 * log(0) = 0)."""
        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        kl = compute_kl_divergence(p, q)
        assert np.isfinite(kl)
        assert kl >= 0.0

    def test_kl_discrete_with_zero_counts(self) -> None:
        """Laplace smoothing should prevent inf when counts are zero."""
        p_counts = np.array([10, 0, 5])
        q_counts = np.array([5, 5, 5])
        kl = compute_kl_discrete(p_counts, q_counts, alpha=1.0)
        assert np.isfinite(kl)
        assert kl >= 0.0


# =====================================================================
# Dimension mismatch errors
# =====================================================================

class TestDimensionMismatch:
    """Test that dimension mismatches raise appropriate errors."""

    def test_different_lengths_raises(self) -> None:
        """p and q of different sizes should raise ValueError."""
        p = np.array([0.5, 0.5])
        q = np.array([1 / 3, 1 / 3, 1 / 3])
        with pytest.raises(ValueError, match="[Ss]hape"):
            compute_kl_divergence(p, q)

    def test_empty_distribution_raises(self) -> None:
        """Empty distribution should raise ValueError."""
        with pytest.raises(ValueError):
            compute_kl_divergence(np.array([]), np.array([]))

    def test_symmetric_kl_dimension_mismatch(self) -> None:
        """JSD should also reject mismatched shapes."""
        with pytest.raises(ValueError):
            symmetric_kl(np.array([0.5, 0.5]), np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_renyi_dimension_mismatch(self) -> None:
        """Rényi divergence should reject mismatched shapes."""
        with pytest.raises(ValueError):
            renyi_divergence(np.array([0.5, 0.5]), np.array([0.25] * 4), alpha=2.0)


# =====================================================================
# Policy-level KL
# =====================================================================

class TestPolicyKL:
    """Test compute_policy_kl with dict-of-dict interface."""

    def test_identical_policies(self) -> None:
        """KL between identical policies is 0."""
        policy = {"s0": {"a0": 0.5, "a1": 0.5}, "s1": {"a0": 1.0}}
        result = compute_policy_kl(policy, policy)
        assert result.total_kl == pytest.approx(0.0, abs=1e-10)
        assert result.is_finite

    def test_different_policies_positive(self) -> None:
        """KL between different policies is positive."""
        policy = {"s0": {"a0": 0.9, "a1": 0.1}}
        reference = {"s0": {"a0": 0.5, "a1": 0.5}}
        result = compute_policy_kl(policy, reference)
        assert result.total_kl > 0.0
        assert result.is_finite

    def test_empty_policy(self) -> None:
        """Empty policy returns zero KL."""
        result = compute_policy_kl({}, {})
        assert result.total_kl == 0.0
