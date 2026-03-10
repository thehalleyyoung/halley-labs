"""Unit tests for usability_oracle.information_theory.entropy.

Tests cover Shannon entropy (uniform, degenerate, custom distributions),
conditional and joint entropy, Rényi and Tsallis generalisations, numerical
stability near zero, entropy rate for Markov chains, and vectorised batch
computation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.information_theory.entropy import (
    batch_entropy,
    batch_renyi_entropy,
    binary_entropy,
    conditional_entropy,
    conditional_entropy_yx,
    cross_entropy,
    differential_entropy_gaussian,
    differential_entropy_gaussian_bits,
    entropy_rate_markov,
    hartley_entropy,
    joint_entropy,
    max_entropy_discrete,
    max_entropy_with_mean,
    min_entropy,
    renyi_entropy,
    shannon_entropy,
    shannon_entropy_bits,
    shannon_entropy_nats,
    tsallis_entropy,
)


# ------------------------------------------------------------------ #
# Shannon entropy — basic
# ------------------------------------------------------------------ #


class TestShannonEntropy:
    """Core Shannon entropy H(X) = -Σ p log p."""

    def test_uniform_binary(self) -> None:
        """H([0.5, 0.5]) = 1 bit."""
        assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)

    def test_uniform_four(self) -> None:
        """H(uniform over 4) = 2 bits."""
        assert shannon_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0)

    def test_uniform_eight(self) -> None:
        """H(uniform over 8) = 3 bits."""
        p = [1 / 8] * 8
        assert shannon_entropy(p) == pytest.approx(3.0)

    def test_degenerate_single_event(self) -> None:
        """H([1.0]) = 0 bits."""
        assert shannon_entropy([1.0]) == pytest.approx(0.0, abs=1e-15)

    def test_degenerate_with_zeros(self) -> None:
        """H([1, 0, 0]) = 0 bits (0 log 0 = 0)."""
        assert shannon_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-15)

    def test_custom_distribution(self) -> None:
        """H([0.75, 0.25]) should be about 0.8113 bits."""
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        assert shannon_entropy([0.75, 0.25]) == pytest.approx(expected, rel=1e-9)

    def test_nats_base(self) -> None:
        """Entropy in nats uses base e."""
        h_nats = shannon_entropy([0.5, 0.5], base=math.e)
        assert h_nats == pytest.approx(math.log(2), rel=1e-9)

    def test_bits_convenience(self) -> None:
        assert shannon_entropy_bits([0.5, 0.5]) == pytest.approx(1.0)

    def test_nats_convenience(self) -> None:
        assert shannon_entropy_nats([0.5, 0.5]) == pytest.approx(math.log(2), rel=1e-9)

    def test_empty_distribution(self) -> None:
        """Empty distribution returns 0."""
        assert shannon_entropy([]) == pytest.approx(0.0)

    def test_numpy_input(self) -> None:
        """Accepts numpy arrays."""
        p = np.array([0.5, 0.5])
        assert shannon_entropy(p) == pytest.approx(1.0)

    def test_non_negative(self) -> None:
        """Entropy is always >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            assert shannon_entropy(p) >= -1e-15

    def test_maximum_at_uniform(self) -> None:
        """Entropy is maximised by the uniform distribution."""
        n = 6
        h_uniform = shannon_entropy([1 / n] * n)
        rng = np.random.default_rng(7)
        for _ in range(20):
            p = rng.dirichlet(np.ones(n))
            assert shannon_entropy(p) <= h_uniform + 1e-10

    def test_base_10(self) -> None:
        """Base 10 gives hartleys."""
        h = shannon_entropy([0.5, 0.5], base=10.0)
        assert h == pytest.approx(math.log10(2), rel=1e-9)


# ------------------------------------------------------------------ #
# Conditional and joint entropy
# ------------------------------------------------------------------ #


class TestConditionalJointEntropy:
    """Tests for H(X|Y), H(Y|X), and H(X,Y)."""

    @pytest.fixture
    def independent_joint(self) -> np.ndarray:
        """Independent X,Y: p(x,y) = p(x)p(y) with uniform marginals."""
        px = np.array([0.5, 0.5])
        py = np.array([0.5, 0.5])
        return np.outer(px, py)

    @pytest.fixture
    def deterministic_joint(self) -> np.ndarray:
        """Perfect correlation: Y = X."""
        return np.array([[0.5, 0.0], [0.0, 0.5]])

    def test_joint_entropy_independent(self, independent_joint: np.ndarray) -> None:
        """H(X,Y) = H(X) + H(Y) for independence."""
        hxy = joint_entropy(independent_joint)
        hx = shannon_entropy(independent_joint.sum(axis=1))
        hy = shannon_entropy(independent_joint.sum(axis=0))
        assert hxy == pytest.approx(hx + hy, rel=1e-9)

    def test_joint_entropy_deterministic(self, deterministic_joint: np.ndarray) -> None:
        """H(X,Y) = H(X) when Y is determined by X."""
        hxy = joint_entropy(deterministic_joint)
        hx = shannon_entropy(deterministic_joint.sum(axis=1))
        assert hxy == pytest.approx(hx, rel=1e-9)

    def test_conditional_entropy_independent(self, independent_joint: np.ndarray) -> None:
        """H(X|Y) = H(X) when X and Y are independent."""
        h_x_given_y = conditional_entropy(independent_joint)
        hx = shannon_entropy(independent_joint.sum(axis=1))
        assert h_x_given_y == pytest.approx(hx, rel=1e-9)

    def test_conditional_entropy_deterministic(self, deterministic_joint: np.ndarray) -> None:
        """H(X|Y) = 0 when Y determines X."""
        h_x_given_y = conditional_entropy(deterministic_joint)
        assert h_x_given_y == pytest.approx(0.0, abs=1e-12)

    def test_conditional_entropy_yx(self, independent_joint: np.ndarray) -> None:
        """H(Y|X) = H(Y) for independent variables."""
        h_y_given_x = conditional_entropy_yx(independent_joint)
        hy = shannon_entropy(independent_joint.sum(axis=0))
        assert h_y_given_x == pytest.approx(hy, rel=1e-9)

    def test_chain_rule(self) -> None:
        """H(X,Y) = H(X) + H(Y|X) — chain rule."""
        pxy = np.array([[0.3, 0.1], [0.2, 0.4]])
        hxy = joint_entropy(pxy)
        hx = shannon_entropy(pxy.sum(axis=1))
        h_y_given_x = conditional_entropy_yx(pxy)
        assert hxy == pytest.approx(hx + h_y_given_x, rel=1e-9)


# ------------------------------------------------------------------ #
# Cross entropy
# ------------------------------------------------------------------ #


class TestCrossEntropy:
    """Tests for H(p, q) = -Σ p log q."""

    def test_same_distribution(self) -> None:
        """H(p, p) = H(p)."""
        p = [0.3, 0.7]
        assert cross_entropy(p, p) == pytest.approx(shannon_entropy(p), rel=1e-9)

    def test_cross_entropy_ge_entropy(self) -> None:
        """H(p, q) >= H(p) (Gibbs' inequality)."""
        p = [0.6, 0.4]
        q = [0.3, 0.7]
        assert cross_entropy(p, q) >= shannon_entropy(p) - 1e-10

    def test_unsupported_gives_inf(self) -> None:
        """H(p, q) = inf when supp(p) not subset supp(q)."""
        assert cross_entropy([0.5, 0.5], [1.0, 0.0]) == float("inf")

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cross_entropy([0.5, 0.5], [0.3, 0.3, 0.4])


# ------------------------------------------------------------------ #
# Rényi entropy
# ------------------------------------------------------------------ #


class TestRenyiEntropy:
    """Tests for H_α(X) = (1/(1-α)) log Σ p^α."""

    def test_alpha_1_equals_shannon(self) -> None:
        """Rényi entropy at α=1 converges to Shannon entropy."""
        p = [0.3, 0.5, 0.2]
        assert renyi_entropy(p, 1.0) == pytest.approx(shannon_entropy(p), rel=1e-9)

    def test_alpha_0_hartley(self) -> None:
        """H_0 = log |support|."""
        p = [0.1, 0.0, 0.3, 0.6]
        assert renyi_entropy(p, 0.0) == pytest.approx(math.log2(3), rel=1e-9)

    def test_min_entropy(self) -> None:
        """H_∞ = -log max p."""
        p = [0.1, 0.6, 0.3]
        assert renyi_entropy(p, float("inf")) == pytest.approx(-math.log2(0.6), rel=1e-9)

    def test_min_entropy_convenience(self) -> None:
        p = [0.1, 0.6, 0.3]
        assert min_entropy(p) == pytest.approx(-math.log2(0.6), rel=1e-9)

    def test_hartley_convenience(self) -> None:
        p = [0.3, 0.0, 0.7]
        assert hartley_entropy(p) == pytest.approx(math.log2(2), rel=1e-9)

    def test_monotone_in_alpha(self) -> None:
        """H_α is non-increasing in α for a fixed distribution."""
        p = [0.1, 0.2, 0.3, 0.4]
        alphas = [0.5, 1.0, 2.0, 5.0, 10.0]
        vals = [renyi_entropy(p, a) for a in alphas]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-10

    def test_uniform_all_orders(self) -> None:
        """For uniform, all Rényi orders give log n."""
        n = 8
        p = [1 / n] * n
        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
            assert renyi_entropy(p, alpha) == pytest.approx(math.log2(n), rel=1e-9)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            renyi_entropy([0.5, 0.5], -1.0)

    def test_nats_base(self) -> None:
        p = [0.5, 0.5]
        h = renyi_entropy(p, 2.0, base=math.e)
        h_bits = renyi_entropy(p, 2.0, base=2.0)
        assert h == pytest.approx(h_bits * math.log(2), rel=1e-9)


# ------------------------------------------------------------------ #
# Tsallis entropy
# ------------------------------------------------------------------ #


class TestTsallisEntropy:
    """Tests for S_q(X) = (1/(q-1))(1 - Σ p^q)."""

    def test_q1_limit_equals_shannon_nats(self) -> None:
        """Tsallis at q=1 → Shannon entropy in nats."""
        p = [0.4, 0.6]
        assert tsallis_entropy(p, 1.0) == pytest.approx(
            shannon_entropy_nats(p), rel=1e-9
        )

    def test_q2_collision_entropy(self) -> None:
        """S_2 = 1 - Σ p²."""
        p = [0.3, 0.7]
        expected = 1.0 - (0.3**2 + 0.7**2)
        assert tsallis_entropy(p, 2.0) == pytest.approx(expected, rel=1e-9)

    def test_uniform_q2(self) -> None:
        """S_2(uniform_n) = 1 - 1/n."""
        n = 5
        p = [1 / n] * n
        assert tsallis_entropy(p, 2.0) == pytest.approx(1.0 - 1.0 / n, rel=1e-9)

    def test_degenerate(self) -> None:
        """S_q([1]) = 0 for all q."""
        assert tsallis_entropy([1.0], 2.0) == pytest.approx(0.0, abs=1e-15)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(10):
            p = rng.dirichlet(np.ones(4))
            assert tsallis_entropy(p, 0.5) >= -1e-15

    def test_invalid_q_raises(self) -> None:
        with pytest.raises(ValueError):
            tsallis_entropy([0.5, 0.5], 0.0)


# ------------------------------------------------------------------ #
# Numerical stability — near-zero probabilities
# ------------------------------------------------------------------ #


class TestNumericalStability:
    """Ensure stability with very small or near-zero probabilities."""

    def test_near_zero_probabilities(self) -> None:
        """Entropy should be finite for tiny probabilities."""
        p = np.array([1e-300, 1.0 - 1e-300])
        h = shannon_entropy(p)
        assert math.isfinite(h)
        assert h >= 0.0

    def test_many_near_zero(self) -> None:
        """Distribution with many near-zero components."""
        n = 100
        p = np.full(n, 1e-10)
        p[0] = 1.0 - (n - 1) * 1e-10
        h = shannon_entropy(p)
        assert math.isfinite(h)
        assert h >= 0.0

    def test_negative_clipped_to_zero(self) -> None:
        """Tiny negative values are clipped to 0."""
        p = np.array([0.5, 0.5, -1e-16])
        h = shannon_entropy(p)
        assert math.isfinite(h)

    def test_binary_entropy_endpoints(self) -> None:
        """Binary entropy at 0 and 1 should be 0."""
        assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-15)
        assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-15)

    def test_binary_entropy_half(self) -> None:
        """h(0.5) = 1 bit."""
        assert binary_entropy(0.5) == pytest.approx(1.0, rel=1e-9)


# ------------------------------------------------------------------ #
# Entropy rate for Markov chains
# ------------------------------------------------------------------ #


class TestEntropyRateMarkov:
    """Tests for H_rate = Σ π(i) H(P(·|i))."""

    def test_iid_process(self) -> None:
        """Entropy rate of i.i.d. process = entropy of the distribution.

        Doubly stochastic matrix with identical rows.
        """
        p = np.array([0.3, 0.7])
        P = np.array([[0.3, 0.7], [0.3, 0.7]])
        rate = entropy_rate_markov(P)
        assert rate == pytest.approx(shannon_entropy(p), rel=1e-6)

    def test_deterministic_chain(self) -> None:
        """Deterministic transitions → entropy rate 0."""
        P = np.array([[0, 1], [1, 0]])  # flip-flop
        rate = entropy_rate_markov(P)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_two_state(self) -> None:
        """Symmetric 2-state chain with P = [[1-p, p], [p, 1-p]]."""
        p = 0.3
        P = np.array([[1 - p, p], [p, 1 - p]])
        rate = entropy_rate_markov(P)
        expected = binary_entropy(p)
        assert rate == pytest.approx(expected, rel=1e-6)

    def test_stationary_provided(self) -> None:
        """When stationary distribution is explicitly given."""
        P = np.array([[0.8, 0.2], [0.4, 0.6]])
        pi = np.array([2.0 / 3, 1.0 / 3])
        rate = entropy_rate_markov(P, stationary=pi)
        expected = pi[0] * shannon_entropy(P[0]) + pi[1] * shannon_entropy(P[1])
        assert rate == pytest.approx(expected, rel=1e-9)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError):
            entropy_rate_markov(np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]]))

    def test_entropy_rate_le_entropy(self) -> None:
        """Entropy rate ≤ entropy of the stationary distribution."""
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        rate = entropy_rate_markov(P)
        # Compute stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
        h_stat = shannon_entropy(pi)
        assert rate <= h_stat + 1e-10


# ------------------------------------------------------------------ #
# Maximum-entropy distributions
# ------------------------------------------------------------------ #


class TestMaxEntropy:
    """Tests for maximum-entropy distribution computation."""

    def test_uniform(self) -> None:
        """Max entropy discrete is uniform."""
        p = max_entropy_discrete(4)
        np.testing.assert_allclose(p, [0.25, 0.25, 0.25, 0.25])

    def test_max_entropy_with_mean(self) -> None:
        """Mean constraint produces exponential family distribution."""
        p = max_entropy_with_mean(5, 2.0)
        assert p.sum() == pytest.approx(1.0, abs=1e-8)
        actual_mean = np.dot(p, np.arange(5))
        assert actual_mean == pytest.approx(2.0, abs=1e-6)

    def test_max_entropy_mean_midpoint_is_uniform(self) -> None:
        """When target mean = midpoint, result is uniform."""
        n = 5
        p = max_entropy_with_mean(n, (n - 1) / 2.0)
        np.testing.assert_allclose(p, np.full(n, 1.0 / n), atol=1e-8)

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError):
            max_entropy_discrete(0)


# ------------------------------------------------------------------ #
# Differential entropy
# ------------------------------------------------------------------ #


class TestDifferentialEntropy:
    """Tests for Gaussian differential entropy."""

    def test_unit_variance(self) -> None:
        """h(N(0,1)) = 0.5 ln(2πe)."""
        expected = 0.5 * math.log(2 * math.pi * math.e)
        assert differential_entropy_gaussian(1.0) == pytest.approx(expected, rel=1e-9)

    def test_bits_conversion(self) -> None:
        h_nats = differential_entropy_gaussian(1.0)
        h_bits = differential_entropy_gaussian_bits(1.0)
        assert h_bits == pytest.approx(h_nats / math.log(2), rel=1e-9)

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            differential_entropy_gaussian(-1.0)

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            differential_entropy_gaussian(0.0)


# ------------------------------------------------------------------ #
# Vectorised batch computation
# ------------------------------------------------------------------ #


class TestBatchEntropy:
    """Tests for batch_entropy and batch_renyi_entropy."""

    def test_batch_matches_individual(self) -> None:
        """Batch computation matches per-row computation."""
        dists = np.array([
            [0.5, 0.5],
            [0.25, 0.75],
            [1.0, 0.0],
        ])
        result = batch_entropy(dists)
        for i in range(3):
            assert result[i] == pytest.approx(
                shannon_entropy(dists[i]), rel=1e-9
            )

    def test_batch_1d(self) -> None:
        """Single distribution (1-D input)."""
        result = batch_entropy(np.array([0.5, 0.5]))
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_batch_renyi_matches_individual(self) -> None:
        dists = np.array([
            [0.5, 0.5],
            [0.2, 0.3, 0.5],
        ], dtype=object)
        # Can't do ragged; use uniform-shape array
        dists2 = np.array([[0.5, 0.5], [0.3, 0.7]])
        result = batch_renyi_entropy(dists2, 2.0)
        for i in range(2):
            assert result[i] == pytest.approx(
                renyi_entropy(dists2[i], 2.0), rel=1e-9
            )

    def test_batch_renyi_alpha1_equals_batch_entropy(self) -> None:
        dists = np.array([[0.4, 0.6], [0.1, 0.9]])
        r1 = batch_renyi_entropy(dists, 1.0)
        se = batch_entropy(dists)
        np.testing.assert_allclose(r1, se, rtol=1e-9)

    def test_batch_non_negative(self) -> None:
        rng = np.random.default_rng(99)
        dists = rng.dirichlet(np.ones(5), size=10)
        result = batch_entropy(dists)
        assert np.all(result >= -1e-15)

    def test_batch_nats(self) -> None:
        dists = np.array([[0.5, 0.5]])
        result_nats = batch_entropy(dists, base=math.e)
        assert result_nats[0] == pytest.approx(math.log(2), rel=1e-9)
