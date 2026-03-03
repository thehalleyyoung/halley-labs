"""Unit tests for cpa.stats.information_theory."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cpa.stats.information_theory import (
    conditional_mutual_information,
    conditional_mutual_information_from_data,
    multi_distribution_jsd,
    multi_distribution_jsd_gaussian,
    mutual_information_discrete,
    mutual_information_from_data,
    mutual_information_gaussian,
    normalized_information_distance,
    normalized_information_distance_gaussian,
    shannon_entropy_discrete,
    shannon_entropy_gaussian,
    shannon_entropy_gaussian_mv,
    transfer_entropy,
)

# ── helpers ──────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)


def _independent_joint(p_x: np.ndarray, p_y: np.ndarray) -> np.ndarray:
    """Build a joint PMF from two independent marginals."""
    return np.outer(p_x, p_y)


def _identity_joint(k: int) -> np.ndarray:
    """Joint PMF where Y = X (perfect correlation)."""
    return np.eye(k) / k


# ── Shannon entropy (discrete) ──────────────────────────────────────


class TestShannonEntropyDiscrete:
    @pytest.mark.parametrize("k", [2, 4, 8, 16])
    def test_uniform_distribution(self, k: int) -> None:
        p = np.ones(k) / k
        expected_nats = math.log(k)
        assert shannon_entropy_discrete(p) == pytest.approx(expected_nats, abs=1e-12)

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_uniform_bits(self, k: int) -> None:
        p = np.ones(k) / k
        expected_bits = math.log2(k)
        assert shannon_entropy_discrete(p, base=2) == pytest.approx(
            expected_bits, abs=1e-12
        )

    def test_degenerate_distribution(self) -> None:
        p = np.array([0.0, 0.0, 1.0, 0.0])
        assert shannon_entropy_discrete(p) == 0.0

    def test_all_zero_returns_zero(self) -> None:
        assert shannon_entropy_discrete(np.zeros(5)) == 0.0

    def test_unnormalized_input(self) -> None:
        p = np.array([2.0, 2.0])
        assert shannon_entropy_discrete(p) == pytest.approx(math.log(2), abs=1e-12)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            shannon_entropy_discrete(np.array([0.5, -0.1, 0.6]))

    def test_nats_less_than_bits(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        h_nats = shannon_entropy_discrete(p)
        h_bits = shannon_entropy_discrete(p, base=2)
        assert h_nats < h_bits  # ln(k) < log2(k) for k >= 3

    def test_binary_entropy(self) -> None:
        p = np.array([0.5, 0.5])
        assert shannon_entropy_discrete(p, base=2) == pytest.approx(1.0, abs=1e-12)


# ── Shannon entropy (Gaussian) ──────────────────────────────────────


class TestShannonEntropyGaussian:
    def test_unit_variance(self) -> None:
        expected = 0.5 * math.log(2 * math.pi * math.e)
        assert shannon_entropy_gaussian(1.0) == pytest.approx(expected, abs=1e-12)

    def test_higher_variance_higher_entropy(self) -> None:
        assert shannon_entropy_gaussian(2.0) > shannon_entropy_gaussian(1.0)

    @pytest.mark.parametrize("d", [1, 2, 5])
    def test_scales_with_dimension(self, d: int) -> None:
        h = shannon_entropy_gaussian(1.0, d=d)
        expected = 0.5 * d * math.log(2 * math.pi * math.e)
        assert h == pytest.approx(expected, abs=1e-12)

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="variance must be > 0"):
            shannon_entropy_gaussian(0.0)

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            shannon_entropy_gaussian(-1.0)

    def test_d_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d must be >= 1"):
            shannon_entropy_gaussian(1.0, d=0)


class TestShannonEntropyGaussianMV:
    def test_identity_covariance(self) -> None:
        d = 3
        cov = np.eye(d)
        expected = 0.5 * d * math.log(2 * math.pi * math.e)
        assert shannon_entropy_gaussian_mv(cov) == pytest.approx(expected, abs=1e-12)

    def test_diagonal_covariance(self) -> None:
        variances = [1.0, 2.0, 3.0]
        cov = np.diag(variances)
        logdet = sum(math.log(v) for v in variances)
        expected = 0.5 * (3 * math.log(2 * math.pi * math.e) + logdet)
        assert shannon_entropy_gaussian_mv(cov) == pytest.approx(expected, abs=1e-12)

    def test_consistent_with_scalar_form(self) -> None:
        var = 2.5
        h_scalar = shannon_entropy_gaussian(var, d=1)
        h_mv = shannon_entropy_gaussian_mv(np.array([[var]]))
        assert h_scalar == pytest.approx(h_mv, abs=1e-12)


# ── Mutual information (discrete) ───────────────────────────────────


class TestMutualInformationDiscrete:
    def test_independent_joint_gives_zero(self) -> None:
        p_x = np.array([0.3, 0.7])
        p_y = np.array([0.4, 0.6])
        joint = _independent_joint(p_x, p_y)
        assert mutual_information_discrete(joint) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_perfect_correlation(self, k: int) -> None:
        joint = _identity_joint(k)
        mi = mutual_information_discrete(joint)
        assert mi == pytest.approx(math.log(k), abs=1e-10)

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            mutual_information_discrete(np.array([0.5, 0.5]))

    def test_bits_base(self) -> None:
        joint = _identity_joint(4)
        mi = mutual_information_discrete(joint, base=2)
        assert mi == pytest.approx(2.0, abs=1e-10)


# ── Mutual information (Gaussian) ───────────────────────────────────


class TestMutualInformationGaussian:
    def test_independent_variables(self) -> None:
        cov = np.diag([1.0, 2.0, 3.0])
        mi = mutual_information_gaussian(cov, [0], [1])
        assert mi == pytest.approx(0.0, abs=1e-12)

    def test_correlated_variables_positive(self) -> None:
        rho = 0.8
        cov = np.array([[1.0, rho], [rho, 1.0]])
        mi = mutual_information_gaussian(cov, [0], [1])
        expected = -0.5 * math.log(1 - rho**2)
        assert mi == pytest.approx(expected, abs=1e-10)
        assert mi > 0

    @pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
    def test_mi_increases_with_correlation(self, rho: float) -> None:
        cov_lo = np.array([[1.0, 0.1], [0.1, 1.0]])
        cov_hi = np.array([[1.0, rho], [rho, 1.0]])
        assert mutual_information_gaussian(cov_hi, [0], [1]) >= mutual_information_gaussian(
            cov_lo, [0], [1]
        )


# ── MI from data ────────────────────────────────────────────────────


class TestMutualInformationFromData:
    def test_gaussian_method_independent(self) -> None:
        n = 5000
        data = _RNG.standard_normal((n, 2))
        mi = mutual_information_from_data(data, [0], [1], method="gaussian")
        assert mi == pytest.approx(0.0, abs=0.05)

    def test_gaussian_method_correlated(self) -> None:
        n = 5000
        rho = 0.7
        z = _RNG.standard_normal((n, 2))
        data = z @ np.linalg.cholesky(np.array([[1, rho], [rho, 1]])).T
        mi = mutual_information_from_data(data, [0], [1], method="gaussian")
        expected = -0.5 * math.log(1 - rho**2)
        assert mi == pytest.approx(expected, abs=0.1)

    def test_binned_method_independent(self) -> None:
        n = 5000
        data = _RNG.standard_normal((n, 2))
        mi = mutual_information_from_data(
            data, [0], [1], method="binned", n_bins=20
        )
        assert mi < 0.1

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            mutual_information_from_data(np.zeros((10, 2)), [0], [1], method="kde")


# ── Conditional mutual information ──────────────────────────────────


class TestConditionalMutualInformation:
    def test_empty_conditioning_reduces_to_mi(self) -> None:
        rho = 0.6
        cov = np.array([[1.0, rho], [rho, 1.0]])
        mi = mutual_information_gaussian(cov, [0], [1])
        cmi = conditional_mutual_information(cov, [0], [1], [])
        assert cmi == pytest.approx(mi, abs=1e-12)

    def test_conditional_independence(self) -> None:
        # X → Z → Y  (X ⊥ Y | Z)
        # X = noise, Z = X + noise, Y = Z + noise
        cov = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
        ])
        cmi = conditional_mutual_information(cov, [0], [2], [1])
        assert cmi == pytest.approx(0.0, abs=1e-10)

    def test_non_conditional_independence_positive(self) -> None:
        # All three mutually correlated — X ⊥̸ Y | Z
        rho = 0.5
        cov = np.array([
            [1.0, rho, 0.1],
            [rho, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        cmi = conditional_mutual_information(cov, [0], [1], [2])
        assert cmi > 0


class TestConditionalMutualInformationFromData:
    def test_ci_from_data(self) -> None:
        # X → Z → Y chain: X ⊥ Y | Z
        n = 5000
        x = _RNG.standard_normal(n)
        z = x + _RNG.standard_normal(n)
        y = z + _RNG.standard_normal(n)
        data = np.column_stack([x, z, y])
        cmi = conditional_mutual_information_from_data(data, [0], [2], [1])
        assert cmi < 0.05


# ── Transfer entropy ────────────────────────────────────────────────


class TestTransferEntropy:
    def test_causal_direction_positive(self) -> None:
        n = 2000
        x = _RNG.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + 0.2 * _RNG.standard_normal()
        te = transfer_entropy(x, y, lag=1, method="gaussian")
        assert te > 0.1

    def test_non_causal_direction_small(self) -> None:
        n = 2000
        x = _RNG.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + 0.2 * _RNG.standard_normal()
        te_reverse = transfer_entropy(y, x, lag=1, method="gaussian")
        te_forward = transfer_entropy(x, y, lag=1, method="gaussian")
        assert te_forward > te_reverse

    def test_independent_series_near_zero(self) -> None:
        n = 2000
        x = _RNG.standard_normal(n)
        y = _RNG.standard_normal(n)
        te = transfer_entropy(x, y, lag=1, method="gaussian")
        assert te < 0.1

    def test_lag_validation(self) -> None:
        with pytest.raises(ValueError, match="lag must be >= 1"):
            transfer_entropy(np.ones(10), np.ones(10), lag=0)

    def test_short_series_raises(self) -> None:
        with pytest.raises(ValueError, match="must exceed lag"):
            transfer_entropy(np.ones(2), np.ones(2), lag=5)

    def test_binned_method_causal(self) -> None:
        n = 3000
        x = _RNG.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.8 * x[t - 1] + 0.2 * _RNG.standard_normal()
        te = transfer_entropy(x, y, lag=1, method="binned", n_bins=15)
        assert te > 0.0


# ── Multi-distribution JSD ──────────────────────────────────────────


class TestMultiDistributionJSD:
    def test_identical_distributions_zero(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        jsd = multi_distribution_jsd([p, p, p])
        assert jsd == pytest.approx(0.0, abs=1e-12)

    def test_different_distributions_positive(self) -> None:
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        jsd = multi_distribution_jsd([p1, p2])
        assert jsd > 0

    def test_two_disjoint_distributions(self) -> None:
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        jsd = multi_distribution_jsd([p1, p2])
        assert jsd == pytest.approx(math.log(2), abs=1e-10)

    def test_custom_weights(self) -> None:
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        jsd_uniform = multi_distribution_jsd([p1, p2])
        jsd_skewed = multi_distribution_jsd(
            [p1, p2], weights=np.array([0.9, 0.1])
        )
        assert jsd_skewed > 0
        assert jsd_skewed != pytest.approx(jsd_uniform, abs=1e-6)

    def test_fewer_than_two_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            multi_distribution_jsd([np.array([1.0])])

    @pytest.mark.parametrize("k", [3, 5])
    def test_jsd_bounded_by_log_k(self, k: int) -> None:
        dists = [np.eye(1, k, i).ravel() for i in range(k)]
        jsd = multi_distribution_jsd(dists)
        assert jsd <= math.log(k) + 1e-10


class TestMultiDistributionJSDGaussian:
    def test_identical_gaussians_near_zero(self) -> None:
        jsd = multi_distribution_jsd_gaussian(
            [0.0, 0.0], [1.0, 1.0], n_points=5000
        )
        assert jsd < 0.01

    def test_separated_gaussians_positive(self) -> None:
        jsd = multi_distribution_jsd_gaussian(
            [0.0, 10.0], [1.0, 1.0], n_points=5000
        )
        assert jsd > 0.1

    def test_fewer_than_two_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            multi_distribution_jsd_gaussian([0.0], [1.0])


# ── Normalised information distance ─────────────────────────────────


class TestNormalizedInformationDistance:
    def test_identical_variables_zero(self) -> None:
        joint = _identity_joint(4)
        nid = normalized_information_distance(joint)
        assert nid == pytest.approx(0.0, abs=1e-10)

    def test_independent_variables_one(self) -> None:
        p_x = np.array([0.5, 0.5])
        p_y = np.array([0.5, 0.5])
        joint = _independent_joint(p_x, p_y)
        nid = normalized_information_distance(joint)
        assert nid == pytest.approx(1.0, abs=1e-10)

    def test_range_zero_one(self) -> None:
        joint = np.array([[0.4, 0.1], [0.1, 0.4]])
        nid = normalized_information_distance(joint)
        assert 0.0 <= nid <= 1.0

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            normalized_information_distance(np.array([0.5, 0.5]))


class TestNormalizedInformationDistanceGaussian:
    def test_independent_nid_one(self) -> None:
        cov = np.diag([1.0, 1.0])
        nid = normalized_information_distance_gaussian(cov, [0], [1])
        assert nid == pytest.approx(1.0, abs=1e-10)

    def test_high_correlation_near_zero(self) -> None:
        rho = 0.99
        cov = np.array([[1.0, rho], [rho, 1.0]])
        nid = normalized_information_distance_gaussian(cov, [0], [1])
        assert nid < 0.3

    def test_nid_decreases_with_correlation(self) -> None:
        nids = []
        for rho in [0.0, 0.5, 0.9]:
            cov = np.array([[1.0, rho], [rho, 1.0]])
            nids.append(normalized_information_distance_gaussian(cov, [0], [1]))
        assert nids[0] > nids[1] > nids[2]
