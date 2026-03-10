"""
Unit tests for usability_oracle.utils.entropy.

Tests cover the Blahut-Arimoto channel capacity algorithm, rate-distortion
function computation, conditional entropy, information gain, and effective
number (perplexity) calculations.  Each test documents the mathematical
invariant or property being verified so that failures are easy to diagnose.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.utils.entropy import (
    channel_capacity,
    conditional_entropy,
    effective_number,
    information_gain,
    rate_distortion,
)


# ===================================================================
# Helpers
# ===================================================================

def _uniform(n: int) -> np.ndarray:
    """Return a uniform distribution of length *n*."""
    return np.full(n, 1.0 / n)


def _identity_channel(n: int) -> np.ndarray:
    """Return a noiseless (identity) transition matrix of size n×n."""
    return np.eye(n)


def _bsc(p: float) -> np.ndarray:
    """Return the transition matrix of a binary symmetric channel with
    crossover probability *p*.

    P(Y|X) = [[1-p, p],
               [p, 1-p]]
    """
    return np.array([[1.0 - p, p], [p, 1.0 - p]])


# ===================================================================
# channel_capacity
# ===================================================================


class TestChannelCapacity:
    """Tests for ``channel_capacity(transition_matrix)``."""

    def test_noiseless_binary_channel(self) -> None:
        """A noiseless 2×2 identity channel has capacity log2(2) = 1 bit."""
        W = _identity_channel(2)
        cap = channel_capacity(W)
        assert cap == pytest.approx(1.0, abs=1e-4)

    def test_noiseless_quaternary_channel(self) -> None:
        """A noiseless 4×4 identity channel has capacity log2(4) = 2 bits."""
        W = _identity_channel(4)
        cap = channel_capacity(W)
        assert cap == pytest.approx(2.0, abs=1e-4)

    def test_noiseless_octal_channel(self) -> None:
        """A noiseless 8×8 identity channel has capacity log2(8) = 3 bits."""
        W = _identity_channel(8)
        cap = channel_capacity(W)
        assert cap == pytest.approx(3.0, abs=1e-4)

    def test_bsc_zero_crossover(self) -> None:
        """BSC with crossover probability 0 is noiseless → capacity = 1 bit."""
        W = _bsc(0.0)
        cap = channel_capacity(W)
        assert cap == pytest.approx(1.0, abs=1e-4)

    def test_bsc_half_crossover(self) -> None:
        """BSC with crossover probability 0.5 is completely noisy → capacity ≈ 0.

        Capacity = 1 − H(0.5) = 1 − 1 = 0.
        """
        W = _bsc(0.5)
        cap = channel_capacity(W)
        assert cap == pytest.approx(0.0, abs=1e-2)

    def test_bsc_known_capacity(self) -> None:
        """BSC with crossover probability p has capacity C = 1 − H(p).

        For p = 0.1:  H(0.1) ≈ 0.469, C ≈ 0.531.
        """
        p = 0.1
        h_p = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        expected = 1.0 - h_p
        cap = channel_capacity(_bsc(p))
        assert cap == pytest.approx(expected, abs=1e-3)

    def test_capacity_is_nonnegative(self) -> None:
        """Channel capacity must always be ≥ 0."""
        rng = np.random.RandomState(0)
        W = rng.dirichlet(np.ones(5), size=4)  # 4×5 random channel
        cap = channel_capacity(W)
        assert cap >= 0.0

    def test_non_square_channel(self) -> None:
        """A non-square channel (3 inputs, 2 outputs) should still converge."""
        W = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        cap = channel_capacity(W)
        assert cap >= 0.0
        assert cap <= math.log2(3)  # upper-bounded by log2(n_inputs)

    def test_single_input_channel(self) -> None:
        """A channel with one input has capacity 0 (no choice)."""
        W = np.array([[0.5, 0.5]])
        cap = channel_capacity(W)
        assert cap == pytest.approx(0.0, abs=1e-6)

    def test_rejects_non_2d(self) -> None:
        """Passing a 1-D array should raise a ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            channel_capacity(np.array([0.5, 0.5]))

    def test_empty_channel_returns_zero(self) -> None:
        """An empty transition matrix should return capacity 0."""
        W = np.zeros((0, 0))
        assert channel_capacity(W) == 0.0


# ===================================================================
# rate_distortion
# ===================================================================


class TestRateDistortion:
    """Tests for ``rate_distortion(source_dist, distortion_matrix, target_rate)``."""

    def test_returns_tuple(self) -> None:
        """rate_distortion should return a (distortion, mapping) tuple."""
        p = _uniform(3)
        D = 1.0 - np.eye(3)  # Hamming distortion
        distortion, mapping = rate_distortion(p, D, target_rate=0.5)
        assert isinstance(distortion, float)
        assert isinstance(mapping, np.ndarray)

    def test_mapping_rows_sum_to_one(self) -> None:
        """The conditional distribution q(x̂|x) rows must each sum to 1."""
        p = _uniform(4)
        D = 1.0 - np.eye(4)
        _, mapping = rate_distortion(p, D, target_rate=1.0)
        row_sums = mapping.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_high_rate_low_distortion(self) -> None:
        """At high target rate, distortion should approach 0."""
        p = _uniform(2)
        D = 1.0 - np.eye(2)
        distortion, _ = rate_distortion(p, D, target_rate=5.0)
        assert distortion < 0.2

    def test_low_rate_high_distortion(self) -> None:
        """At very low target rate, distortion should be larger."""
        p = _uniform(2)
        D = 1.0 - np.eye(2)
        d_low, _ = rate_distortion(p, D, target_rate=0.01)
        d_high, _ = rate_distortion(p, D, target_rate=5.0)
        assert d_low >= d_high - 1e-6

    def test_distortion_nonnegative(self) -> None:
        """Achieved distortion must be ≥ 0."""
        p = np.array([0.7, 0.3])
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        distortion, _ = rate_distortion(p, D, target_rate=0.5)
        assert distortion >= -1e-8


# ===================================================================
# conditional_entropy
# ===================================================================


class TestConditionalEntropy:
    """Tests for ``conditional_entropy(joint)``."""

    def test_deterministic_joint(self) -> None:
        """If Y is a function of X, H(Y|X) = 0.

        Joint with all mass on the diagonal → conditional entropy is 0.
        """
        joint = np.diag([0.5, 0.5])
        assert conditional_entropy(joint) == pytest.approx(0.0, abs=1e-8)

    def test_independent_joint(self) -> None:
        """If X and Y are independent, H(Y|X) = H(Y).

        Joint = p(x) p(y) for uniform marginals → H(Y|X) = log2(n).
        """
        n = 4
        joint = np.full((n, n), 1.0 / (n * n))
        cond_h = conditional_entropy(joint)
        assert cond_h == pytest.approx(math.log2(n), abs=1e-4)

    def test_nonnegative(self) -> None:
        """Conditional entropy is always ≥ 0."""
        rng = np.random.RandomState(7)
        joint = rng.dirichlet(np.ones(9)).reshape(3, 3)
        assert conditional_entropy(joint) >= -1e-10

    def test_at_most_marginal_entropy(self) -> None:
        """H(Y|X) ≤ H(Y), with equality iff independence."""
        rng = np.random.RandomState(42)
        joint = rng.dirichlet(np.ones(6)).reshape(2, 3)
        cond_h = conditional_entropy(joint)
        p_y = joint.sum(axis=0)
        h_y = float(-np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0])))
        assert cond_h <= h_y + 1e-8

    def test_rejects_1d_input(self) -> None:
        """Should raise ValueError for a 1-D array."""
        with pytest.raises(ValueError, match="2-D"):
            conditional_entropy(np.array([0.5, 0.5]))

    def test_zero_joint_returns_zero(self) -> None:
        """A zero joint distribution should yield entropy 0."""
        joint = np.zeros((3, 3))
        assert conditional_entropy(joint) == 0.0


# ===================================================================
# information_gain
# ===================================================================


class TestInformationGain:
    """Tests for ``information_gain(prior, posterior)``."""

    def test_no_change_yields_zero(self) -> None:
        """If prior == posterior, information gain is 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert information_gain(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_posterior_more_certain(self) -> None:
        """Gain is positive when posterior has lower entropy than prior."""
        prior = np.array([0.5, 0.5])
        posterior = np.array([0.9, 0.1])
        assert information_gain(prior, posterior) > 0.0

    def test_negative_when_posterior_less_certain(self) -> None:
        """Gain is negative when posterior is more uncertain."""
        prior = np.array([0.9, 0.1])
        posterior = np.array([0.5, 0.5])
        assert information_gain(prior, posterior) < 0.0

    def test_maximum_gain_from_uniform_to_deterministic(self) -> None:
        """Going from uniform to deterministic yields max gain = log2(n)."""
        n = 8
        prior = _uniform(n)
        posterior = np.zeros(n)
        posterior[0] = 1.0
        ig = information_gain(prior, posterior)
        assert ig == pytest.approx(math.log2(n), abs=1e-6)


# ===================================================================
# effective_number
# ===================================================================


class TestEffectiveNumber:
    """Tests for ``effective_number(probs)``."""

    def test_uniform_distribution(self) -> None:
        """For a uniform distribution over n elements, effective number = n."""
        for n in [2, 4, 8, 16]:
            p = _uniform(n)
            assert effective_number(p) == pytest.approx(float(n), abs=1e-4)

    def test_single_element(self) -> None:
        """If all probability is on one element, effective number = 1."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        assert effective_number(p) == pytest.approx(1.0, abs=1e-8)

    def test_result_at_least_one(self) -> None:
        """Effective number should always be ≥ 1 for a valid distribution."""
        p = np.array([0.99, 0.005, 0.005])
        assert effective_number(p) >= 1.0

    def test_at_most_n(self) -> None:
        """Effective number should be ≤ n (number of categories)."""
        n = 10
        rng = np.random.RandomState(99)
        p = rng.dirichlet(np.ones(n))
        assert effective_number(p) <= n + 1e-8

    def test_two_equal_elements(self) -> None:
        """If probability is split equally between 2 of n bins, perplexity = 2."""
        p = np.array([0.5, 0.5, 0.0, 0.0])
        assert effective_number(p) == pytest.approx(2.0, abs=1e-6)

    def test_monotonicity_with_spread(self) -> None:
        """Spreading probability more uniformly should increase effective number.

        A more uniform distribution has higher entropy, hence higher perplexity.
        """
        p_concentrated = np.array([0.9, 0.05, 0.05])
        p_spread = np.array([0.4, 0.3, 0.3])
        assert effective_number(p_spread) > effective_number(p_concentrated)

    def test_empty_distribution(self) -> None:
        """An empty probability vector has effective number = 2^0 = 1."""
        p = np.array([])
        assert effective_number(p) == pytest.approx(1.0, abs=1e-8)
