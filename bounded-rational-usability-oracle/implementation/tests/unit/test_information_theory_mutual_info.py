"""Unit tests for usability_oracle.information_theory.mutual_information.

Tests cover mutual information computation, KL divergence for known
distributions, Jensen-Shannon divergence (symmetry, boundedness),
f-divergences, and edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.information_theory.mutual_information import (
    batch_kl_divergence,
    chi_squared_divergence,
    f_divergence,
    hellinger_distance,
    jensen_shannon_distance,
    jensen_shannon_divergence,
    kl_divergence,
    kl_divergence_bits,
    kl_divergence_nats,
    mutual_information,
    mutual_information_full,
    normalized_mutual_information,
    pointwise_mutual_information,
    symmetrized_kl,
    total_correlation,
    total_variation_distance,
)
from usability_oracle.information_theory.entropy import shannon_entropy


# ------------------------------------------------------------------ #
# Mutual information I(X; Y)
# ------------------------------------------------------------------ #


class TestMutualInformation:
    """Tests for mutual information from joint distributions."""

    def test_independent_mi_zero(self) -> None:
        """I(X;Y) = 0 when X and Y are independent."""
        px = np.array([0.5, 0.5])
        py = np.array([0.5, 0.5])
        joint = np.outer(px, py)
        assert mutual_information(joint) == pytest.approx(0.0, abs=1e-12)

    def test_perfect_correlation(self) -> None:
        """I(X;Y) = H(X) when Y = X."""
        joint = np.array([[0.5, 0.0], [0.0, 0.5]])
        mi = mutual_information(joint)
        hx = shannon_entropy([0.5, 0.5])
        assert mi == pytest.approx(hx, rel=1e-9)

    def test_mi_non_negative(self) -> None:
        """MI is always >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            joint = rng.dirichlet(np.ones(9)).reshape(3, 3)
            assert mutual_information(joint) >= -1e-12

    def test_mi_le_marginal_entropy(self) -> None:
        """I(X;Y) <= min(H(X), H(Y))."""
        joint = np.array([[0.3, 0.1], [0.2, 0.4]])
        mi = mutual_information(joint)
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        assert mi <= min(shannon_entropy(px), shannon_entropy(py)) + 1e-10

    def test_symmetric(self) -> None:
        """I(X;Y) = I(Y;X) — swapping joint axes gives same MI."""
        joint = np.array([[0.3, 0.1], [0.2, 0.4]])
        mi1 = mutual_information(joint)
        mi2 = mutual_information(joint.T)
        assert mi1 == pytest.approx(mi2, rel=1e-9)


class TestMutualInformationFull:
    """Tests for the full MI decomposition."""

    def test_decomposition_identity(self) -> None:
        """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
        joint = np.array([[0.3, 0.05], [0.15, 0.5]])
        result = mutual_information_full(joint)
        assert result.mutual_info_bits == pytest.approx(
            result.entropy_x_bits + result.entropy_y_bits - result.joint_entropy_bits,
            rel=1e-9,
        )

    def test_conditional_entropies(self) -> None:
        """H(Y|X) = H(X,Y) - H(X)."""
        joint = np.array([[0.3, 0.05], [0.15, 0.5]])
        result = mutual_information_full(joint)
        assert result.conditional_entropy_y_given_x_bits == pytest.approx(
            result.joint_entropy_bits - result.entropy_x_bits, rel=1e-9,
        )

    def test_normalized_mi(self) -> None:
        """NMI in [0, 1]."""
        joint = np.array([[0.3, 0.05], [0.15, 0.5]])
        result = mutual_information_full(joint)
        assert 0.0 <= result.normalized <= 1.0 + 1e-10


class TestNormalizedMI:
    """Tests for normalised MI variants."""

    @pytest.fixture
    def joint(self) -> np.ndarray:
        return np.array([[0.3, 0.05], [0.15, 0.5]])

    def test_min_variant(self, joint: np.ndarray) -> None:
        nmi = normalized_mutual_information(joint, variant="min")
        assert 0.0 <= nmi <= 1.0 + 1e-10

    def test_max_variant(self, joint: np.ndarray) -> None:
        nmi = normalized_mutual_information(joint, variant="max")
        assert 0.0 <= nmi <= 1.0 + 1e-10

    def test_sqrt_variant(self, joint: np.ndarray) -> None:
        nmi = normalized_mutual_information(joint, variant="sqrt")
        assert 0.0 <= nmi <= 1.0 + 1e-10

    def test_sum_variant(self, joint: np.ndarray) -> None:
        nmi = normalized_mutual_information(joint, variant="sum")
        assert 0.0 <= nmi <= 1.0 + 1e-10

    def test_joint_variant(self, joint: np.ndarray) -> None:
        nmi = normalized_mutual_information(joint, variant="joint")
        assert 0.0 <= nmi <= 1.0 + 1e-10

    def test_unknown_variant_raises(self, joint: np.ndarray) -> None:
        with pytest.raises(ValueError):
            normalized_mutual_information(joint, variant="foo")


# ------------------------------------------------------------------ #
# KL divergence
# ------------------------------------------------------------------ #


class TestKLDivergence:
    """Tests for D_KL(p ‖ q)."""

    def test_same_distribution(self) -> None:
        """D_KL(p ‖ p) = 0."""
        p = [0.3, 0.7]
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        """D_KL([0.5,0.5] ‖ [0.25,0.75]) in bits."""
        p = [0.5, 0.5]
        q = [0.25, 0.75]
        expected = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
        assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-9)

    def test_non_negative(self) -> None:
        """D_KL >= 0 (Gibbs' inequality)."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            assert kl_divergence(p, q) >= -1e-12

    def test_asymmetric(self) -> None:
        """D_KL(p ‖ q) ≠ D_KL(q ‖ p) in general."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=0.01)

    def test_infinite_when_unsupported(self) -> None:
        """D_KL = ∞ when supp(p) ⊄ supp(q)."""
        p = [0.5, 0.5]
        q = [1.0, 0.0]
        assert kl_divergence(p, q) == float("inf")

    def test_bits_convenience(self) -> None:
        p = [0.5, 0.5]
        q = [0.25, 0.75]
        assert kl_divergence_bits(p, q) == pytest.approx(kl_divergence(p, q), rel=1e-12)

    def test_nats_convenience(self) -> None:
        p = [0.5, 0.5]
        q = [0.25, 0.75]
        kl_nats = kl_divergence_nats(p, q)
        kl_bits = kl_divergence_bits(p, q)
        assert kl_nats == pytest.approx(kl_bits * math.log(2), rel=1e-9)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_symmetrized_kl(self) -> None:
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        skl = symmetrized_kl(p, q)
        expected = (kl_divergence(p, q) + kl_divergence(q, p)) / 2.0
        assert skl == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------------ #
# Jensen-Shannon divergence
# ------------------------------------------------------------------ #


class TestJensenShannonDivergence:
    """Tests for JSD(p ‖ q)."""

    def test_same_distribution_zero(self) -> None:
        """JSD(p ‖ p) = 0."""
        p = [0.3, 0.7]
        assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric(self) -> None:
        """JSD(p ‖ q) = JSD(q ‖ p)."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        assert jensen_shannon_divergence(p, q) == pytest.approx(
            jensen_shannon_divergence(q, p), rel=1e-12,
        )

    def test_bounded_by_log2(self) -> None:
        """JSD ∈ [0, 1] in bits (base 2)."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        jsd = jensen_shannon_divergence(p, q)
        assert 0.0 <= jsd <= 1.0 + 1e-10

    def test_maximum_for_disjoint(self) -> None:
        """JSD is maximised for disjoint-support distributions."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        assert jensen_shannon_divergence(p, q) == pytest.approx(1.0, rel=1e-6)

    def test_jsd_le_kl(self) -> None:
        """JSD(p ‖ q) ≤ min(D_KL(p‖q), D_KL(q‖p))."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        jsd = jensen_shannon_divergence(p, q)
        assert jsd <= kl_divergence(p, q) + 1e-10
        assert jsd <= kl_divergence(q, p) + 1e-10

    def test_js_distance(self) -> None:
        """JS distance = sqrt(JSD) and is a metric."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        jsd = jensen_shannon_divergence(p, q)
        assert jensen_shannon_distance(p, q) == pytest.approx(
            math.sqrt(jsd), rel=1e-12,
        )

    def test_triangle_inequality(self) -> None:
        """JS distance satisfies triangle inequality."""
        p = [0.2, 0.8]
        q = [0.5, 0.5]
        r = [0.9, 0.1]
        d_pq = jensen_shannon_distance(p, q)
        d_qr = jensen_shannon_distance(q, r)
        d_pr = jensen_shannon_distance(p, r)
        assert d_pr <= d_pq + d_qr + 1e-10


# ------------------------------------------------------------------ #
# f-divergences
# ------------------------------------------------------------------ #


class TestFDivergences:
    """Tests for general f-divergences and special cases."""

    def test_kl_via_f(self) -> None:
        """KL divergence as an f-divergence with f(t) = t log t."""
        p = [0.4, 0.6]
        q = [0.5, 0.5]
        kl_f = f_divergence(p, q, lambda t: t * math.log(t) if t > 0 else 0.0)
        kl_direct = kl_divergence(p, q, base=math.e)
        assert kl_f == pytest.approx(kl_direct, rel=1e-6)

    def test_chi_squared(self) -> None:
        """χ² divergence = Σ (p-q)²/q."""
        p = [0.4, 0.6]
        q = [0.5, 0.5]
        expected = (0.4 - 0.5) ** 2 / 0.5 + (0.6 - 0.5) ** 2 / 0.5
        assert chi_squared_divergence(p, q) == pytest.approx(expected, rel=1e-9)

    def test_chi_squared_same_zero(self) -> None:
        p = [0.3, 0.7]
        assert chi_squared_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_chi_squared_unsupported_inf(self) -> None:
        assert chi_squared_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")

    def test_hellinger_same_zero(self) -> None:
        p = [0.3, 0.7]
        assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_hellinger_range(self) -> None:
        """Hellinger distance ∈ [0, 1]."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        h = hellinger_distance(p, q)
        assert 0.0 <= h <= 1.0 + 1e-10

    def test_hellinger_symmetric(self) -> None:
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        assert hellinger_distance(p, q) == pytest.approx(
            hellinger_distance(q, p), rel=1e-12,
        )

    def test_total_variation_same_zero(self) -> None:
        p = [0.3, 0.7]
        assert total_variation_distance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_total_variation_range(self) -> None:
        """TV ∈ [0, 1]."""
        p = [1.0, 0.0]
        q = [0.0, 1.0]
        assert total_variation_distance(p, q) == pytest.approx(1.0, rel=1e-9)

    def test_pinsker_inequality(self) -> None:
        """TV(p,q) ≤ sqrt(0.5 * D_KL(p‖q)) — Pinsker's inequality (nats)."""
        p = [0.3, 0.7]
        q = [0.6, 0.4]
        tv = total_variation_distance(p, q)
        kl_nats = kl_divergence(p, q, base=math.e)
        assert tv <= math.sqrt(0.5 * kl_nats) + 1e-10


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


class TestMIEdgeCases:
    """Edge cases for mutual information and divergence functions."""

    def test_mi_single_element(self) -> None:
        """1×1 joint → MI = 0."""
        joint = np.array([[1.0]])
        assert mutual_information(joint) == pytest.approx(0.0, abs=1e-12)

    def test_kl_identical_uniform(self) -> None:
        """D_KL of two identical uniform distributions is 0."""
        p = [0.25, 0.25, 0.25, 0.25]
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_pointwise_mi_shape(self) -> None:
        """PMI matrix has same shape as joint."""
        joint = np.array([[0.3, 0.1], [0.2, 0.4]])
        pmi = pointwise_mutual_information(joint)
        assert pmi.shape == joint.shape

    def test_pointwise_mi_mean_equals_mi(self) -> None:
        """Expected PMI = MI."""
        joint = np.array([[0.3, 0.1], [0.2, 0.4]])
        pmi = pointwise_mutual_information(joint)
        mi = mutual_information(joint)
        expected_pmi = float(np.sum(joint * pmi))
        assert expected_pmi == pytest.approx(mi, rel=1e-6)

    def test_batch_kl_matches_individual(self) -> None:
        p = np.array([[0.5, 0.5], [0.3, 0.7]])
        q = np.array([[0.4, 0.6], [0.6, 0.4]])
        result = batch_kl_divergence(p, q)
        for i in range(2):
            assert result[i] == pytest.approx(kl_divergence(p[i], q[i]), rel=1e-9)

    def test_batch_kl_inf(self) -> None:
        p = np.array([[0.5, 0.5]])
        q = np.array([[1.0, 0.0]])
        result = batch_kl_divergence(p, q)
        assert result[0] == float("inf")

    def test_total_correlation_independent(self) -> None:
        """TC = 0 for independent variables."""
        joint = np.outer([0.5, 0.5], [0.5, 0.5])
        assert total_correlation(joint) == pytest.approx(0.0, abs=1e-12)

    def test_total_correlation_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        joint = rng.dirichlet(np.ones(4)).reshape(2, 2)
        assert total_correlation(joint) >= -1e-12
