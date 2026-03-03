"""Unit tests for cpa.core.mechanism_distance.

Tests cover:
  - MechanismDistanceComputer: KL divergence, sqrt(JSD), multivariate variants,
    conditional distances, Monte-Carlo JSD, multi-distribution JSD,
    mechanism identity test, pairwise distances, caching.
  - DistanceMatrix: construction, summary statistics, clustering, serialization.
  - Helper functions: _ensure_positive, _regularize_covariance, _symmetrize.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sp_stats

from cpa.core.mechanism_distance import (
    DistanceMatrix,
    MechanismDistanceComputer,
    _ensure_positive,
    _regularize_covariance,
    _symmetrize,
)
from cpa.core.scm import StructuralCausalModel
from cpa.core.mccm import MultiContextCausalModel
from cpa.core.types import Context


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def computer() -> MechanismDistanceComputer:
    """Standard MechanismDistanceComputer with fixed seed."""
    return MechanismDistanceComputer(n_mc_samples=5000, seed=42)


@pytest.fixture
def computer_no_cache() -> MechanismDistanceComputer:
    """Computer with caching disabled."""
    return MechanismDistanceComputer(n_mc_samples=5000, seed=42, cache_enabled=False)


@pytest.fixture
def simple_scm() -> StructuralCausalModel:
    """3-variable chain: X0 -> X1 -> X2."""
    adj = np.array([
        [0, 0.5, 0],
        [0, 0, 0.8],
        [0, 0, 0],
    ])
    return StructuralCausalModel(
        adjacency_matrix=(adj != 0).astype(float),
        variable_names=["X0", "X1", "X2"],
        regression_coefficients=adj,
        residual_variances=np.array([1.0, 1.0, 1.0]),
        sample_size=200,
    )


@pytest.fixture
def shifted_scm() -> StructuralCausalModel:
    """Same structure as simple_scm but different coefficients."""
    adj = np.array([
        [0, 1.5, 0],
        [0, 0, 0.2],
        [0, 0, 0],
    ])
    return StructuralCausalModel(
        adjacency_matrix=(adj != 0).astype(float),
        variable_names=["X0", "X1", "X2"],
        regression_coefficients=adj,
        residual_variances=np.array([2.0, 0.5, 3.0]),
        sample_size=200,
    )


@pytest.fixture
def identical_scm(simple_scm) -> StructuralCausalModel:
    """Exact copy of simple_scm."""
    adj = simple_scm.regression_coefficients.copy()
    return StructuralCausalModel(
        adjacency_matrix=simple_scm.adjacency_matrix.copy(),
        variable_names=list(simple_scm.variable_names),
        regression_coefficients=adj,
        residual_variances=simple_scm.residual_variances.copy(),
        sample_size=simple_scm.sample_size,
    )


@pytest.fixture
def distance_matrix_3x3() -> DistanceMatrix:
    """Simple 3x3 distance matrix."""
    mat = np.array([
        [0.0, 0.3, 0.7],
        [0.3, 0.0, 0.5],
        [0.7, 0.5, 0.0],
    ])
    return DistanceMatrix(mat, labels=["ctx_a", "ctx_b", "ctx_c"], variable_idx=0)


@pytest.fixture
def mccm_two_contexts(simple_scm, shifted_scm) -> MultiContextCausalModel:
    """MCCM with two contexts."""
    mccm = MultiContextCausalModel(mode="intersection")
    mccm.add_context(Context(id="ctx1"), simple_scm)
    mccm.add_context(Context(id="ctx2"), shifted_scm)
    return mccm


# ===================================================================
# Helper function tests
# ===================================================================


class TestEnsurePositive:
    def test_positive_passthrough(self):
        assert _ensure_positive(1.0, "x") == 1.0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            _ensure_positive(-1.0, "x")

    def test_very_small_clamped(self):
        with pytest.warns(UserWarning, match="below minimum"):
            result = _ensure_positive(1e-20, "x")
        assert result > 0

    def test_zero_clamped(self):
        with pytest.warns(UserWarning, match="below minimum"):
            result = _ensure_positive(0.0, "x")
        assert result > 0


class TestRegularizeCovariance:
    def test_identity_unchanged(self):
        I = np.eye(3)
        result = _regularize_covariance(I)
        assert_allclose(result, I, atol=1e-8)

    def test_singular_becomes_pd(self):
        S = np.array([[1, 1], [1, 1]], dtype=float)  # rank-1
        result = _regularize_covariance(S)
        eigvals = np.linalg.eigvalsh(result)
        assert eigvals.min() > 0, "Should be positive-definite after regularization"

    def test_empty_matrix(self):
        empty = np.zeros((0, 0))
        result = _regularize_covariance(empty)
        assert result.shape == (0, 0)


class TestSymmetrize:
    def test_already_symmetric(self):
        M = np.array([[1, 2], [2, 3]], dtype=float)
        assert_allclose(_symmetrize(M), M)

    def test_asymmetric(self):
        M = np.array([[1, 3], [1, 4]], dtype=float)
        result = _symmetrize(M)
        assert_allclose(result, result.T)
        assert_allclose(result, np.array([[1, 2], [2, 4]], dtype=float))


# ===================================================================
# Constructor tests
# ===================================================================


class TestMechanismDistanceComputerInit:
    def test_default_construction(self):
        c = MechanismDistanceComputer()
        assert c.n_mc_samples == 50_000
        assert c.cache_enabled is True

    def test_custom_params(self):
        c = MechanismDistanceComputer(n_mc_samples=200, seed=7, regularization=1e-6)
        assert c.n_mc_samples == 200

    def test_min_mc_samples(self):
        with pytest.raises(ValueError, match="n_mc_samples must be >= 100"):
            MechanismDistanceComputer(n_mc_samples=50)

    def test_exactly_100_ok(self):
        c = MechanismDistanceComputer(n_mc_samples=100)
        assert c.n_mc_samples == 100


# ===================================================================
# KL divergence — univariate Gaussian
# ===================================================================


class TestKLGaussian:
    def test_identical_distributions(self, computer):
        """KL(P, P) = 0."""
        assert computer.kl_gaussian(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_nonnegative(self, computer):
        """KL(P, Q) >= 0 for all P, Q."""
        kl = computer.kl_gaussian(0.0, 1.0, 1.0, 2.0)
        assert kl >= 0

    @pytest.mark.parametrize("mu1,s1sq,mu2,s2sq", [
        (0, 1, 1, 1),
        (0, 1, 0, 2),
        (2, 3, -1, 0.5),
    ])
    def test_known_values(self, computer, mu1, s1sq, mu2, s2sq):
        """Compare against manually computed KL."""
        expected = (
            0.5 * np.log(s2sq / s1sq)
            + (s1sq + (mu1 - mu2) ** 2) / (2.0 * s2sq)
            - 0.5
        )
        result = computer.kl_gaussian(mu1, s1sq, mu2, s2sq)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_asymmetry(self, computer):
        """KL(P, Q) != KL(Q, P) in general."""
        kl_pq = computer.kl_gaussian(0, 1, 2, 3)
        kl_qp = computer.kl_gaussian(2, 3, 0, 1)
        assert kl_pq != pytest.approx(kl_qp, abs=1e-6)

    def test_negative_variance_raises(self, computer):
        with pytest.raises(ValueError, match="non-negative"):
            computer.kl_gaussian(0, -1, 0, 1)

    def test_negative_variance_q_raises(self, computer):
        with pytest.raises(ValueError, match="non-negative"):
            computer.kl_gaussian(0, 1, 0, -1)


# ===================================================================
# sqrt(JSD) — univariate Gaussian
# ===================================================================


class TestSqrtJSDGaussian:
    def test_identical_zero(self, computer):
        """Identical distributions → distance 0."""
        d = computer.sqrt_jsd_gaussian(0.0, 1.0, 0.0, 1.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_very_different_close_to_one(self, computer):
        """Very separated distributions → distance close to 1."""
        d = computer.sqrt_jsd_gaussian(0.0, 0.01, 100.0, 0.01)
        assert d > 0.95

    def test_symmetry(self, computer):
        """sqrt(JSD) is symmetric: d(P,Q) == d(Q,P)."""
        d1 = computer.sqrt_jsd_gaussian(0.0, 1.0, 2.0, 3.0)
        d2 = computer.sqrt_jsd_gaussian(2.0, 3.0, 0.0, 1.0)
        assert d1 == pytest.approx(d2, abs=1e-14)

    def test_range_01(self, computer):
        """Result must be in [0, 1]."""
        d = computer.sqrt_jsd_gaussian(0.0, 1.0, 5.0, 2.0)
        assert 0 <= d <= 1

    @pytest.mark.parametrize("mu1,sigma1,mu2,sigma2", [
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (0, 1, 0, 2),
        (-5, 0.1, 5, 0.1),
    ])
    def test_nonnegative(self, computer, mu1, sigma1, mu2, sigma2):
        d = computer.sqrt_jsd_gaussian(mu1, sigma1, mu2, sigma2)
        assert d >= 0

    def test_triangle_inequality(self, computer):
        """sqrt(JSD) satisfies triangle inequality."""
        d_ab = computer.sqrt_jsd_gaussian(0, 1, 2, 1)
        d_bc = computer.sqrt_jsd_gaussian(2, 1, 4, 1)
        d_ac = computer.sqrt_jsd_gaussian(0, 1, 4, 1)
        assert d_ac <= d_ab + d_bc + 1e-10

    def test_same_mean_different_variance(self, computer):
        """Same mean, different variance → positive distance."""
        d = computer.sqrt_jsd_gaussian(0, 1, 0, 3)
        assert d > 0

    def test_same_variance_different_mean(self, computer):
        """Same variance, different mean → positive distance."""
        d = computer.sqrt_jsd_gaussian(0, 1, 5, 1)
        assert d > 0


# ===================================================================
# KL divergence — multivariate Gaussian
# ===================================================================


class TestKLMultivariateGaussian:
    def test_identical(self, computer):
        mu = np.zeros(3)
        Sigma = np.eye(3)
        kl = computer.kl_multivariate_gaussian(mu, Sigma, mu, Sigma)
        assert kl == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self, computer):
        mu1 = np.array([0, 0])
        mu2 = np.array([1, 2])
        S1 = np.eye(2)
        S2 = np.array([[2, 0.5], [0.5, 1]])
        kl = computer.kl_multivariate_gaussian(mu1, S1, mu2, S2)
        assert kl >= -1e-10

    def test_known_univariate_correspondence(self, computer):
        """1-D multivariate KL should match univariate KL."""
        mu1, mu2 = np.array([0.0]), np.array([1.0])
        S1, S2 = np.array([[1.0]]), np.array([[2.0]])
        kl_mv = computer.kl_multivariate_gaussian(mu1, S1, mu2, S2)
        kl_uv = computer.kl_gaussian(0.0, 1.0, 1.0, 2.0)
        assert kl_mv == pytest.approx(kl_uv, rel=1e-6)

    def test_zero_dim(self, computer):
        mu = np.array([])
        S = np.zeros((0, 0))
        assert computer.kl_multivariate_gaussian(mu, S, mu, S) == 0.0

    def test_shape_mismatch_raises(self, computer):
        mu1 = np.zeros(2)
        mu2 = np.zeros(3)
        S1 = np.eye(2)
        S2 = np.eye(3)
        with pytest.raises(ValueError, match="same shape"):
            computer.kl_multivariate_gaussian(mu1, S1, mu2, S2)

    def test_different_means_identity_cov(self, computer):
        """KL between N(mu1, I) and N(mu2, I) = 0.5 * ||mu1-mu2||^2."""
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([1.0, 2.0])
        I = np.eye(2)
        kl = computer.kl_multivariate_gaussian(mu1, I, mu2, I)
        expected = 0.5 * np.sum((mu1 - mu2) ** 2)
        assert kl == pytest.approx(expected, rel=1e-6)


# ===================================================================
# sqrt(JSD) — multivariate Gaussian
# ===================================================================


class TestSqrtJSDMultivariateGaussian:
    def test_identical_zero(self, computer):
        mu = np.zeros(2)
        S = np.eye(2)
        d = computer.sqrt_jsd_multivariate_gaussian(mu, S, mu, S)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_symmetry(self, computer):
        mu1 = np.array([0, 0])
        mu2 = np.array([3, 1])
        S1 = np.eye(2)
        S2 = np.array([[2, 0.3], [0.3, 1]])
        d1 = computer.sqrt_jsd_multivariate_gaussian(mu1, S1, mu2, S2)
        d2 = computer.sqrt_jsd_multivariate_gaussian(mu2, S2, mu1, S1)
        assert d1 == pytest.approx(d2, abs=1e-12)

    def test_range_01(self, computer):
        mu1 = np.zeros(3)
        mu2 = np.array([10, 10, 10])
        S1 = np.eye(3) * 0.01
        S2 = np.eye(3) * 0.01
        d = computer.sqrt_jsd_multivariate_gaussian(mu1, S1, mu2, S2)
        assert 0 <= d <= 1

    def test_different_covariances(self, computer):
        mu = np.zeros(2)
        S1 = np.eye(2)
        S2 = np.eye(2) * 5
        d = computer.sqrt_jsd_multivariate_gaussian(mu, S1, mu, S2)
        assert d > 0

    def test_zero_dim(self, computer):
        mu = np.array([])
        S = np.zeros((0, 0))
        assert computer.sqrt_jsd_multivariate_gaussian(mu, S, mu, S) == 0.0

    def test_univariate_consistency(self, computer):
        """1-D multivariate JSD should be close to univariate JSD."""
        mu1, mu2 = np.array([0.0]), np.array([2.0])
        S1, S2 = np.array([[1.0]]), np.array([[4.0]])
        d_mv = computer.sqrt_jsd_multivariate_gaussian(mu1, S1, mu2, S2)
        d_uv = computer.sqrt_jsd_gaussian(0.0, 1.0, 2.0, 2.0)
        assert d_mv == pytest.approx(d_uv, rel=1e-6)


# ===================================================================
# sqrt(JSD) — conditional
# ===================================================================


class TestSqrtJSDConditional:
    def test_same_coefficients_same_variance(self, computer):
        """Identical mechanisms → distance 0."""
        c = np.array([0.5, 0.3])
        d = computer.sqrt_jsd_conditional(c, 1.0, c, 1.0)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_same_coefficients_different_variance(self, computer):
        """Same coefficients, different noise → positive distance."""
        c = np.array([0.5, 0.3])
        d = computer.sqrt_jsd_conditional(c, 1.0, c, 5.0)
        assert d > 0

    def test_different_coefficients(self, computer):
        c1 = np.array([0.5, 0.3])
        c2 = np.array([2.0, -1.0])
        d = computer.sqrt_jsd_conditional(c1, 1.0, c2, 1.0)
        assert d > 0

    def test_no_parents(self, computer):
        """No parents: reduces to comparing noise distributions."""
        d = computer.sqrt_jsd_conditional(
            np.array([]), 1.0, np.array([]), 4.0,
            intercept1=0.0, intercept2=0.0,
        )
        d_ref = computer.sqrt_jsd_gaussian(0.0, 1.0, 0.0, 2.0)
        assert d == pytest.approx(d_ref, abs=1e-8)

    def test_different_parent_sizes_zero_padded(self, computer):
        """Different parent set sizes should be handled via zero-padding."""
        c1 = np.array([1.0, 0.5])
        c2 = np.array([1.0])
        d = computer.sqrt_jsd_conditional(c1, 1.0, c2, 1.0)
        assert d >= 0

    def test_with_parent_covariance(self, computer):
        c = np.array([1.0])
        parent_cov = np.array([[4.0]])
        d = computer.sqrt_jsd_conditional(
            c, 1.0, c, 1.0, parent_cov=parent_cov
        )
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_intercepts_matter(self, computer):
        c = np.array([])
        d = computer.sqrt_jsd_conditional(
            c, 1.0, c, 1.0, intercept1=0.0, intercept2=10.0
        )
        assert d > 0

    def test_range_01(self, computer):
        c1 = np.array([5.0, -3.0])
        c2 = np.array([-5.0, 3.0])
        d = computer.sqrt_jsd_conditional(c1, 0.1, c2, 0.1)
        assert 0 <= d <= 1


# ===================================================================
# sqrt(JSD) — conditional Monte-Carlo
# ===================================================================


class TestSqrtJSDConditionalMC:
    def test_identical_mechanisms(self, computer):
        c = np.array([0.5])
        parent_samples = np.random.default_rng(0).normal(size=(500, 1))
        d = computer.sqrt_jsd_conditional_mc(c, 1.0, c, 1.0, parent_samples)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_different_mechanisms_positive(self, computer):
        c1 = np.array([1.0])
        c2 = np.array([5.0])
        parent_samples = np.random.default_rng(0).normal(size=(200, 1))
        d = computer.sqrt_jsd_conditional_mc(c1, 1.0, c2, 1.0, parent_samples)
        assert d > 0

    def test_empty_parent_samples(self, computer):
        c1 = np.array([1.0])
        c2 = np.array([1.0])
        parent_samples = np.zeros((0, 1))
        d = computer.sqrt_jsd_conditional_mc(
            c1, 1.0, c2, 1.0, parent_samples,
            intercept1=0.0, intercept2=5.0,
        )
        # Falls back to sqrt_jsd_gaussian with intercepts
        assert d > 0


# ===================================================================
# sqrt(JSD) — Monte-Carlo generic
# ===================================================================


class TestSqrtJSDMonteCarlo:
    def test_identical_distributions(self, computer):
        rng = np.random.default_rng(123)
        samples = rng.normal(0, 1, size=(5000, 1))
        dist = sp_stats.norm(0, 1)
        d = computer.sqrt_jsd_monte_carlo(
            dist.logpdf, dist.logpdf, samples, samples
        )
        assert d == pytest.approx(0.0, abs=0.05)

    def test_different_distributions(self, computer):
        rng = np.random.default_rng(123)
        p = sp_stats.norm(0, 1)
        q = sp_stats.norm(5, 1)
        samples_p = rng.normal(0, 1, size=(5000, 1))
        samples_q = rng.normal(5, 1, size=(5000, 1))
        d = computer.sqrt_jsd_monte_carlo(
            p.logpdf, q.logpdf, samples_p, samples_q
        )
        assert d > 0.3

    def test_range_01(self, computer):
        rng = np.random.default_rng(0)
        p = sp_stats.norm(0, 1)
        q = sp_stats.norm(10, 0.1)
        samples_p = rng.normal(0, 1, size=(3000, 1))
        samples_q = rng.normal(10, 0.1, size=(3000, 1))
        d = computer.sqrt_jsd_monte_carlo(
            p.logpdf, q.logpdf, samples_p, samples_q
        )
        assert 0 <= d <= 1


# ===================================================================
# Multi-distribution JSD
# ===================================================================


class TestMultiDistributionJSD:
    def test_two_identical(self, computer):
        mu = np.array([0.0])
        S = np.array([[1.0]])
        d = computer.multi_distribution_jsd([(mu, S), (mu, S)])
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_empty_list(self, computer):
        assert computer.multi_distribution_jsd([]) == 0.0

    def test_single_distribution(self, computer):
        mu = np.array([0.0])
        S = np.array([[1.0]])
        assert computer.multi_distribution_jsd([(mu, S)]) == 0.0

    def test_two_different_positive(self, computer):
        d1 = (np.array([0.0]), np.array([[1.0]]))
        d2 = (np.array([5.0]), np.array([[1.0]]))
        d = computer.multi_distribution_jsd([d1, d2])
        assert d > 0

    def test_k_way_bounded(self, computer):
        """K-way JSD with uniform weights should be in [0,1]."""
        dists = [
            (np.array([float(k)]), np.array([[1.0]]))
            for k in range(5)
        ]
        d = computer.multi_distribution_jsd(dists)
        assert 0 <= d <= 1

    def test_custom_weights(self, computer):
        d1 = (np.array([0.0]), np.array([[1.0]]))
        d2 = (np.array([10.0]), np.array([[1.0]]))
        w = np.array([0.9, 0.1])
        d = computer.multi_distribution_jsd([d1, d2], weights=w)
        assert d > 0

    def test_weights_not_summing_to_one_raises(self, computer):
        d1 = (np.array([0.0]), np.array([[1.0]]))
        with pytest.raises(ValueError, match="sum to 1"):
            computer.multi_distribution_jsd([d1, d1], weights=np.array([0.5, 0.6]))

    def test_multivariate(self, computer):
        d1 = (np.zeros(2), np.eye(2))
        d2 = (np.array([3.0, 3.0]), np.eye(2))
        d3 = (np.array([-3.0, -3.0]), np.eye(2))
        d = computer.multi_distribution_jsd([d1, d2, d3])
        assert 0 < d <= 1


# ===================================================================
# Mechanism identity test
# ===================================================================


class TestMechanismIdentityTest:
    def test_identical_scms(self, computer, simple_scm, identical_scm):
        """Identical SCMs → identical=True for each variable."""
        for var_idx in range(3):
            result = computer.mechanism_identity_test(
                simple_scm, identical_scm, var_idx, n_bootstrap=100,
            )
            assert result["identical"] is True
            assert result["distance"] == pytest.approx(0.0, abs=1e-6)
            assert "p_value" in result
            assert "details" in result

    def test_very_different_scms(self, computer, simple_scm, shifted_scm):
        """Very different coefficients → likely not identical for some variable."""
        result = computer.mechanism_identity_test(
            simple_scm, shifted_scm, 1, n_bootstrap=100,
        )
        assert result["distance"] > 0
        assert "coeff_p_value" in result
        assert "var_p_value" in result

    def test_var_idx_out_of_range(self, computer, simple_scm, identical_scm):
        with pytest.raises(ValueError, match="out of range"):
            computer.mechanism_identity_test(simple_scm, identical_scm, 5)

    def test_different_parent_sets(self, computer):
        """Different parent sets → identical=False."""
        adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        scm1 = StructuralCausalModel(
            adjacency_matrix=adj1, regression_coefficients=adj1,
            residual_variances=np.ones(3), sample_size=100,
        )
        scm2 = StructuralCausalModel(
            adjacency_matrix=adj2, regression_coefficients=adj2,
            residual_variances=np.ones(3), sample_size=100,
        )
        # Variable 1: parents differ (scm1 has parent 0, scm2 has none)
        result = computer.mechanism_identity_test(scm1, scm2, 1, n_bootstrap=50)
        assert result["identical"] is False
        assert result["details"]["reason"] == "different_parent_sets"

    def test_result_dict_keys(self, computer, simple_scm, identical_scm):
        result = computer.mechanism_identity_test(
            simple_scm, identical_scm, 0, n_bootstrap=50,
        )
        expected_keys = {"identical", "p_value", "distance", "coeff_p_value", "var_p_value", "details"}
        assert expected_keys.issubset(result.keys())


# ===================================================================
# Pairwise mechanism distances
# ===================================================================


class TestPairwiseMechanismDistances:
    def test_returns_distance_matrix(self, computer, mccm_two_contexts):
        dm = computer.pairwise_mechanism_distances(mccm_two_contexts, 0)
        assert isinstance(dm, DistanceMatrix)
        assert dm.K == 2

    def test_diagonal_zero(self, computer, mccm_two_contexts):
        dm = computer.pairwise_mechanism_distances(mccm_two_contexts, 0)
        for i in range(dm.K):
            assert dm.matrix[i, i] == pytest.approx(0.0)

    def test_symmetry(self, computer, mccm_two_contexts):
        dm = computer.pairwise_mechanism_distances(mccm_two_contexts, 1)
        assert_allclose(dm.matrix, dm.matrix.T)

    def test_identical_contexts_zero_distance(self, computer, simple_scm):
        mccm = MultiContextCausalModel(mode="intersection")
        mccm.add_context(Context(id="a"), simple_scm)
        adj = simple_scm.regression_coefficients.copy()
        scm_copy = StructuralCausalModel(
            adjacency_matrix=simple_scm.adjacency_matrix.copy(),
            variable_names=list(simple_scm.variable_names),
            regression_coefficients=adj,
            residual_variances=simple_scm.residual_variances.copy(),
            sample_size=simple_scm.sample_size,
        )
        mccm.add_context(Context(id="b"), scm_copy)
        dm = computer.pairwise_mechanism_distances(mccm, 0)
        assert dm.matrix[0, 1] == pytest.approx(0.0, abs=1e-8)


# ===================================================================
# Cache behavior
# ===================================================================


class TestCache:
    def test_cache_hit(self, computer):
        d1 = computer.sqrt_jsd_gaussian(0.0, 1.0, 2.0, 1.0)
        d2 = computer.sqrt_jsd_gaussian(0.0, 1.0, 2.0, 1.0)
        assert d1 == d2
        assert len(computer._cache) > 0

    def test_clear_cache(self, computer):
        computer.sqrt_jsd_gaussian(0.0, 1.0, 2.0, 1.0)
        assert len(computer._cache) > 0
        computer.clear_cache()
        assert len(computer._cache) == 0

    def test_no_cache_when_disabled(self, computer_no_cache):
        computer_no_cache.sqrt_jsd_gaussian(0.0, 1.0, 2.0, 1.0)
        assert len(computer_no_cache._cache) == 0

    def test_cache_key_deterministic(self, computer):
        k1 = computer._cache_key("gauss1d", 0.0, 1.0)
        k2 = computer._cache_key("gauss1d", 0.0, 1.0)
        assert k1 == k2

    def test_cache_key_distinct_for_different_args(self, computer):
        k1 = computer._cache_key("gauss1d", 0.0, 1.0)
        k2 = computer._cache_key("gauss1d", 0.0, 2.0)
        assert k1 != k2

    def test_cache_key_with_array(self, computer):
        a = np.array([1.0, 2.0])
        k1 = computer._cache_key("mv", a)
        k2 = computer._cache_key("mv", a.copy())
        assert k1 == k2


# ===================================================================
# DistanceMatrix
# ===================================================================


class TestDistanceMatrix:
    def test_construction(self, distance_matrix_3x3):
        dm = distance_matrix_3x3
        assert dm.K == 3
        assert dm.labels == ["ctx_a", "ctx_b", "ctx_c"]
        assert dm.variable_idx == 0

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            DistanceMatrix(np.zeros((2, 3)))

    def test_wrong_label_count_raises(self):
        with pytest.raises(ValueError, match="labels"):
            DistanceMatrix(np.zeros((2, 2)), labels=["a"])

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            DistanceMatrix(np.zeros(5))

    def test_default_labels(self):
        dm = DistanceMatrix(np.zeros((2, 2)))
        assert dm.labels == ["0", "1"]

    def test_mean(self, distance_matrix_3x3):
        vals = distance_matrix_3x3.upper_triangle
        assert distance_matrix_3x3.mean() == pytest.approx(np.mean(vals))

    def test_max(self, distance_matrix_3x3):
        assert distance_matrix_3x3.max() == pytest.approx(0.7)

    def test_min(self, distance_matrix_3x3):
        assert distance_matrix_3x3.min() == pytest.approx(0.3)

    def test_median(self, distance_matrix_3x3):
        assert distance_matrix_3x3.median() == pytest.approx(0.5)

    def test_std(self, distance_matrix_3x3):
        vals = distance_matrix_3x3.upper_triangle
        assert distance_matrix_3x3.std() == pytest.approx(np.std(vals))

    def test_percentile(self, distance_matrix_3x3):
        assert 0.3 <= distance_matrix_3x3.percentile(50) <= 0.7

    def test_threshold_count(self, distance_matrix_3x3):
        assert distance_matrix_3x3.threshold_count(0.4) == 2  # 0.5 and 0.7

    def test_distance_by_label(self, distance_matrix_3x3):
        assert distance_matrix_3x3.distance("ctx_a", "ctx_b") == pytest.approx(0.3)
        assert distance_matrix_3x3.distance("ctx_a", "ctx_c") == pytest.approx(0.7)

    def test_symmetry_of_matrix(self, distance_matrix_3x3):
        assert_allclose(
            distance_matrix_3x3.matrix,
            distance_matrix_3x3.matrix.T,
        )

    def test_summary_keys(self, distance_matrix_3x3):
        s = distance_matrix_3x3.summary()
        expected_keys = {"mean", "median", "std", "min", "max", "p25", "p75", "p95", "n_pairs"}
        assert expected_keys == set(s.keys())
        assert s["n_pairs"] == 3

    def test_nearest_neighbor(self, distance_matrix_3x3):
        label, dist = distance_matrix_3x3.nearest_neighbor("ctx_a")
        assert label == "ctx_b"
        assert dist == pytest.approx(0.3)

    def test_k_nearest_neighbors(self, distance_matrix_3x3):
        neighbors = distance_matrix_3x3.k_nearest_neighbors("ctx_a", k=2)
        assert len(neighbors) == 2
        assert neighbors[0][0] == "ctx_b"
        assert neighbors[1][0] == "ctx_c"

    def test_submatrix(self, distance_matrix_3x3):
        sub = distance_matrix_3x3.submatrix(["ctx_a", "ctx_c"])
        assert sub.K == 2
        assert sub.distance("ctx_a", "ctx_c") == pytest.approx(0.7)

    def test_filter_by_threshold(self, distance_matrix_3x3):
        pairs = distance_matrix_3x3.filter_by_threshold(0.4)
        assert len(pairs) == 2
        # Should be sorted descending by distance
        assert pairs[0][2] >= pairs[1][2]

    def test_to_dict_roundtrip(self, distance_matrix_3x3):
        d = distance_matrix_3x3.to_dict()
        restored = DistanceMatrix.from_dict(d)
        assert restored == distance_matrix_3x3

    def test_heatmap_data(self, distance_matrix_3x3):
        data = distance_matrix_3x3.heatmap_data()
        assert "matrix" in data
        assert "labels" in data
        assert len(data["labels"]) == 3

    def test_repr(self, distance_matrix_3x3):
        r = repr(distance_matrix_3x3)
        assert "DistanceMatrix" in r
        assert "K=3" in r

    def test_eq(self):
        m = np.array([[0, 1], [1, 0]], dtype=float)
        dm1 = DistanceMatrix(m, labels=["a", "b"])
        dm2 = DistanceMatrix(m.copy(), labels=["a", "b"])
        assert dm1 == dm2

    def test_neq_different_labels(self):
        m = np.array([[0, 1], [1, 0]], dtype=float)
        dm1 = DistanceMatrix(m, labels=["a", "b"])
        dm2 = DistanceMatrix(m.copy(), labels=["x", "y"])
        assert dm1 != dm2


class TestDistanceMatrixClustering:
    def test_hierarchical_with_n_clusters(self, distance_matrix_3x3):
        labels = distance_matrix_3x3.hierarchical_clustering(n_clusters=2)
        assert len(labels) == 3
        assert len(set(labels)) == 2

    def test_hierarchical_with_threshold(self, distance_matrix_3x3):
        labels = distance_matrix_3x3.hierarchical_clustering(threshold=0.4)
        assert len(labels) == 3

    def test_hierarchical_default(self, distance_matrix_3x3):
        labels = distance_matrix_3x3.hierarchical_clustering()
        assert len(labels) == 3

    def test_single_element(self):
        dm = DistanceMatrix(np.zeros((1, 1)), labels=["only"])
        labels = dm.hierarchical_clustering(n_clusters=1)
        assert len(labels) == 1
        assert labels[0] == 0


class TestDistanceMatrixEdgeCases:
    def test_empty_matrix_stats(self):
        dm = DistanceMatrix(np.zeros((1, 1)), labels=["a"])
        assert dm.mean() == 0.0
        assert dm.max() == 0.0
        assert dm.min() == 0.0

    def test_all_zeros(self):
        dm = DistanceMatrix(np.zeros((3, 3)), labels=["a", "b", "c"])
        assert dm.mean() == 0.0


# ===================================================================
# Edge cases for MechanismDistanceComputer
# ===================================================================


class TestMechanismDistanceEdgeCases:
    def test_kl_gaussian_very_small_variance(self, computer):
        """Very small (but positive) variance should not crash."""
        kl = computer.kl_gaussian(0, 1e-13, 0, 1.0)
        assert np.isfinite(kl)

    def test_multivariate_near_singular(self, computer):
        """Near-singular covariance should be regularized, not crash."""
        mu = np.zeros(2)
        S_good = np.eye(2)
        S_bad = np.array([[1, 0.9999999], [0.9999999, 1]])
        d = computer.sqrt_jsd_multivariate_gaussian(mu, S_good, mu, S_bad)
        assert np.isfinite(d)

    def test_conditional_large_coefficients(self, computer):
        c1 = np.array([100.0, -100.0])
        c2 = np.array([-100.0, 100.0])
        d = computer.sqrt_jsd_conditional(c1, 1.0, c2, 1.0)
        assert np.isfinite(d)
        assert 0 <= d <= 1

    def test_multi_jsd_dimension_mismatch_raises(self, computer):
        d1 = (np.array([0.0]), np.array([[1.0]]))
        d2 = (np.array([0.0, 1.0]), np.eye(2))
        with pytest.raises(ValueError, match="same dimension"):
            computer.multi_distribution_jsd([d1, d2])


# ===================================================================
# Batch mechanism distances
# ===================================================================


class TestBatchMechanismDistances:
    def test_batch_all_variables(self, computer, mccm_two_contexts):
        result = computer.batch_mechanism_distances(mccm_two_contexts)
        assert len(result) == 3  # 3 variables
        for vi, dm in result.items():
            assert isinstance(dm, DistanceMatrix)
            assert dm.K == 2

    def test_batch_specific_variables(self, computer, mccm_two_contexts):
        result = computer.batch_mechanism_distances(
            mccm_two_contexts, var_indices=[0, 2]
        )
        assert set(result.keys()) == {0, 2}


# ===================================================================
# Structural divergence
# ===================================================================


class TestStructuralDivergence:
    def test_identical_dags(self, computer):
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        assert computer.structural_divergence(adj, adj) == pytest.approx(0.0)

    def test_completely_different(self, computer):
        adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        adj2 = np.zeros((3, 3))
        d = computer.structural_divergence(adj1, adj2)
        assert d > 0

    def test_empty_dags(self, computer):
        adj = np.zeros((0, 0))
        assert computer.structural_divergence(adj, adj) == 0.0

    def test_symmetry(self, computer):
        adj1 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
        adj2 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        d1 = computer.structural_divergence(adj1, adj2)
        d2 = computer.structural_divergence(adj2, adj1)
        # Not necessarily symmetric due to addition/deletion weights,
        # but with default equal weights it should be close
        assert d1 == pytest.approx(d2, abs=1e-10)


# ===================================================================
# Combined mechanism divergence
# ===================================================================


class TestCombinedMechanismDivergence:
    def test_identical_scms(self, computer, simple_scm, identical_scm):
        result = computer.combined_mechanism_divergence(simple_scm, identical_scm)
        assert result["combined"] == pytest.approx(0.0, abs=1e-6)
        assert result["structural"] == pytest.approx(0.0, abs=1e-6)
        assert result["parametric"] == pytest.approx(0.0, abs=1e-6)

    def test_result_keys(self, computer, simple_scm, shifted_scm):
        result = computer.combined_mechanism_divergence(simple_scm, shifted_scm)
        assert set(result.keys()) == {"combined", "structural", "parametric", "per_variable"}

    def test_combined_is_weighted_sum(self, computer, simple_scm, shifted_scm):
        result = computer.combined_mechanism_divergence(
            simple_scm, shifted_scm,
            structural_weight=0.3, parametric_weight=0.7,
        )
        expected = 0.3 * result["structural"] + 0.7 * result["parametric"]
        assert result["combined"] == pytest.approx(expected, rel=1e-10)
