"""
Unit tests for usability_oracle.utils.math.

Tests cover numerically stable mathematical utilities: log2_safe,
Shannon entropy, KL divergence, mutual information, log-sum-exp,
softmax, distribution normalisation, total variation distance,
Jensen-Shannon divergence, and Wasserstein distance.  Tests emphasise
mathematical properties (non-negativity, symmetry, metric axioms)
as well as known-value checks against hand-computed results.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.utils.math import (
    entropy,
    jensen_shannon_divergence,
    kl_divergence,
    log2_safe,
    log_sum_exp,
    mutual_information,
    normalize_distribution,
    softmax,
    total_variation_distance,
    wasserstein_distance,
)


# ===================================================================
# Helpers
# ===================================================================

def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _deterministic(n: int, k: int = 0) -> np.ndarray:
    """Distribution with all mass on element *k*."""
    p = np.zeros(n)
    p[k] = 1.0
    return p


# ===================================================================
# log2_safe
# ===================================================================


class TestLog2Safe:
    """Tests for ``log2_safe(x)``."""

    def test_positive_values(self) -> None:
        """log2_safe returns correct values: log2(8)=3, log2(1)=0, log2(0.5)=-1."""
        assert log2_safe(8.0) == pytest.approx(3.0)
        assert log2_safe(1.0) == pytest.approx(0.0)
        assert log2_safe(0.5) == pytest.approx(-1.0)

    def test_zero_and_negative_return_zero(self) -> None:
        """The implementation returns 0 for x ≤ 0 instead of raising."""
        assert log2_safe(0.0) == 0.0
        assert log2_safe(-5.0) == 0.0


# ===================================================================
# entropy
# ===================================================================


class TestEntropy:
    """Tests for ``entropy(probs)``."""

    def test_uniform_is_log2_n(self) -> None:
        """Entropy of a uniform distribution over n symbols is log2(n)."""
        for n in [2, 4, 8]:
            p = _uniform(n)
            assert entropy(p) == pytest.approx(math.log2(n), abs=1e-8)

    def test_deterministic_is_zero(self) -> None:
        """Entropy of a deterministic distribution is 0."""
        p = _deterministic(5)
        assert entropy(p) == pytest.approx(0.0, abs=1e-10)

    def test_binary_entropy_and_nonnegativity(self) -> None:
        """H([0.25, 0.75]) should match hand-computed value, and H ≥ 0 always."""
        p = np.array([0.25, 0.75])
        expected = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
        assert entropy(p) == pytest.approx(expected, abs=1e-8)
        rng = np.random.RandomState(1)
        assert entropy(rng.dirichlet(np.ones(7))) >= 0.0
        assert entropy(np.array([])) == 0.0


# ===================================================================
# kl_divergence
# ===================================================================


class TestKLDivergence:
    """Tests for ``kl_divergence(p, q)``."""

    def test_identical_distributions(self) -> None:
        """D_KL(P || P) = 0 for any valid P."""
        p = np.array([0.3, 0.7])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative_and_asymmetric(self) -> None:
        """KL divergence is always ≥ 0, and D_KL(P || Q) ≠ D_KL(Q || P)."""
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        d_pq = kl_divergence(p, q)
        d_qp = kl_divergence(q, p)
        assert d_pq >= -1e-10
        assert d_qp >= -1e-10
        assert d_pq != pytest.approx(d_qp, abs=1e-6)

    def test_known_value(self) -> None:
        """D_KL([0.5, 0.5] || [0.25, 0.75]) should match hand computation."""
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
        assert kl_divergence(p, q) == pytest.approx(expected, abs=1e-6)

    def test_mismatched_lengths_raises(self) -> None:
        """Distributions of different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            kl_divergence(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


# ===================================================================
# mutual_information
# ===================================================================


class TestMutualInformation:
    """Tests for ``mutual_information(joint)``."""

    def test_independent_variables(self) -> None:
        """If X ⊥ Y, mutual information is 0.  Joint = p(x)p(y) for uniform."""
        n = 3
        joint = np.full((n, n), 1.0 / (n * n))
        assert mutual_information(joint) == pytest.approx(0.0, abs=1e-6)

    def test_perfect_dependence(self) -> None:
        """If Y = X (diagonal joint), MI = H(X) = log2(n)."""
        n = 4
        joint = np.diag(_uniform(n))
        mi = mutual_information(joint)
        assert mi == pytest.approx(math.log2(n), abs=1e-4)

    def test_nonnegative_and_rejects_1d(self) -> None:
        """MI is always ≥ 0, and a 1-D array should raise ValueError."""
        rng = np.random.RandomState(5)
        joint = rng.dirichlet(np.ones(12)).reshape(3, 4)
        assert mutual_information(joint) >= -1e-10
        with pytest.raises(ValueError, match="2-D"):
            mutual_information(np.array([0.5, 0.5]))
        assert mutual_information(np.zeros((3, 3))) == 0.0


# ===================================================================
# log_sum_exp
# ===================================================================


class TestLogSumExp:
    """Tests for ``log_sum_exp(values)``."""

    def test_matches_naive(self) -> None:
        """Result should match the naïve log(sum(exp(v))) for moderate values."""
        v = np.array([1.0, 2.0, 3.0])
        expected = math.log(sum(math.exp(x) for x in v))
        assert log_sum_exp(v) == pytest.approx(expected, abs=1e-10)

    def test_numerically_stable_large_values(self) -> None:
        """Should not overflow for large values."""
        v = np.array([1000.0, 1001.0, 1002.0])
        result = log_sum_exp(v)
        assert math.isfinite(result)
        expected = 1002.0 + math.log(1.0 + math.exp(-1.0) + math.exp(-2.0))
        assert result == pytest.approx(expected, abs=1e-8)

    def test_edge_cases(self) -> None:
        """Single element returns itself; empty returns -inf; all -inf returns -inf."""
        assert log_sum_exp(np.array([5.0])) == pytest.approx(5.0, abs=1e-10)
        assert log_sum_exp(np.array([])) == -math.inf
        assert log_sum_exp(np.array([-math.inf, -math.inf])) == -math.inf


# ===================================================================
# softmax
# ===================================================================


class TestSoftmax:
    """Tests for ``softmax(values, temperature)``."""

    def test_sums_to_one(self) -> None:
        """Softmax output must sum to 1."""
        v = np.array([1.0, 2.0, 3.0])
        result = softmax(v)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_high_temperature_approaches_uniform(self) -> None:
        """At very high temperature, softmax → uniform distribution."""
        v = np.array([1.0, 5.0, 10.0])
        result = softmax(v, temperature=1e6)
        expected_uniform = 1.0 / len(v)
        np.testing.assert_allclose(result, expected_uniform, atol=1e-4)

    def test_low_temperature_concentrates_on_max(self) -> None:
        """At very low temperature, softmax places all mass on the argmax."""
        v = np.array([1.0, 5.0, 10.0])
        result = softmax(v, temperature=1e-3)
        assert result[2] > 0.99

    def test_equal_inputs_give_uniform(self) -> None:
        """If all inputs are equal, output is uniform."""
        v = np.array([3.0, 3.0, 3.0, 3.0])
        result = softmax(v)
        np.testing.assert_allclose(result, 0.25, atol=1e-10)

    def test_zero_temperature_raises(self) -> None:
        """Temperature ≤ 0 should raise ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            softmax(np.array([1.0, 2.0]), temperature=0.0)

    def test_preserves_ordering(self) -> None:
        """Larger input logits should map to larger probabilities."""
        v = np.array([1.0, 3.0, 2.0])
        result = softmax(v)
        assert result[1] > result[2] > result[0]


# ===================================================================
# normalize_distribution
# ===================================================================


class TestNormalizeDistribution:
    """Tests for ``normalize_distribution(values)``."""

    def test_sums_to_one_and_preserves_ratios(self) -> None:
        """Output sums to 1 and preserves relative proportions."""
        v = np.array([2.0, 4.0, 4.0])
        result = normalize_distribution(v)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)
        assert result[1] / result[0] == pytest.approx(2.0, abs=1e-8)

    def test_zero_input_returns_uniform(self) -> None:
        """An all-zero input should return a uniform distribution."""
        result = normalize_distribution(np.zeros(4))
        np.testing.assert_allclose(result, 0.25, atol=1e-10)

    def test_negative_values_clipped(self) -> None:
        """Negative values should be clipped to 0 before normalising."""
        v = np.array([-1.0, 2.0, 3.0])
        result = normalize_distribution(v)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)


# ===================================================================
# total_variation_distance
# ===================================================================


class TestTotalVariationDistance:
    """Tests for ``total_variation_distance(p, q)``."""

    def test_identical_distributions(self) -> None:
        """TV distance between identical distributions is 0."""
        p = np.array([0.3, 0.7])
        assert total_variation_distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_disjoint_support(self) -> None:
        """Distributions with disjoint support have TV = 1."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert total_variation_distance(p, q) == pytest.approx(1.0, abs=1e-10)

    def test_range_zero_to_one(self) -> None:
        """TV distance is always in [0, 1] for probability distributions."""
        rng = np.random.RandomState(8)
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        tv = total_variation_distance(p, q)
        assert 0.0 <= tv <= 1.0 + 1e-10

    def test_symmetric(self) -> None:
        """TV(P, Q) == TV(Q, P)."""
        p = np.array([0.6, 0.4])
        q = np.array([0.3, 0.7])
        assert total_variation_distance(p, q) == pytest.approx(
            total_variation_distance(q, p), abs=1e-10
        )

    def test_mismatched_lengths_raises(self) -> None:
        """Distributions of different lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            total_variation_distance(np.array([0.5, 0.5]), np.array([1.0]))


# ===================================================================
# jensen_shannon_divergence
# ===================================================================


class TestJensenShannonDivergence:
    """Tests for ``jensen_shannon_divergence(p, q)``."""

    def test_identical_distributions(self) -> None:
        """JSD between identical distributions is 0."""
        p = np.array([0.25, 0.75])
        assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self) -> None:
        """JSD(P, Q) == JSD(Q, P) — it is a symmetrised divergence."""
        p = np.array([0.9, 0.1])
        q = np.array([0.2, 0.8])
        assert jensen_shannon_divergence(p, q) == pytest.approx(
            jensen_shannon_divergence(q, p), abs=1e-10
        )

    def test_nonnegative(self) -> None:
        """JSD is always ≥ 0."""
        rng = np.random.RandomState(12)
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        assert jensen_shannon_divergence(p, q) >= -1e-10

    def test_upper_bound(self) -> None:
        """JSD ≤ log2(2) = 1 for two distributions.

        The maximum occurs for disjoint supports.
        """
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert jsd <= 1.0 + 1e-6

    def test_known_value_disjoint(self) -> None:
        """JSD of two disjoint Dirac deltas should be exactly 1 bit (= log2(2))."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert jsd == pytest.approx(1.0, abs=1e-4)


# ===================================================================
# wasserstein_distance
# ===================================================================


class TestWassersteinDistance:
    """Tests for ``wasserstein_distance(p, q)``."""

    def test_identical_distributions(self) -> None:
        """Wasserstein distance between identical distributions is 0."""
        p = np.array([0.25, 0.5, 0.25])
        assert wasserstein_distance(p, p) == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self) -> None:
        """Wasserstein distance is always ≥ 0."""
        rng = np.random.RandomState(20)
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        assert wasserstein_distance(p, q) >= -1e-10

    def test_symmetric(self) -> None:
        """W(P, Q) == W(Q, P) — it is a metric."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        assert wasserstein_distance(p, q) == pytest.approx(
            wasserstein_distance(q, p), abs=1e-8
        )

    def test_dirac_deltas_at_endpoints(self) -> None:
        """Moving all mass from one end to the other should give a
        positive distance proportional to the number of bins."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0, 1.0])
        d = wasserstein_distance(p, q)
        assert d > 0.0

    def test_triangle_inequality(self) -> None:
        """W(P, R) ≤ W(P, Q) + W(Q, R) — metric triangle inequality."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.5, 0.3])
        r = np.array([0.1, 0.1, 0.8])
        d_pr = wasserstein_distance(p, r)
        d_pq = wasserstein_distance(p, q)
        d_qr = wasserstein_distance(q, r)
        assert d_pr <= d_pq + d_qr + 1e-8

    def test_with_cost_matrix(self) -> None:
        """When a custom cost matrix is provided, the distance should
        respect it.  Using a zero cost matrix should yield distance 0."""
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        cost = np.zeros((2, 2))
        assert wasserstein_distance(p, q, cost_matrix=cost) == pytest.approx(
            0.0, abs=1e-8
        )
