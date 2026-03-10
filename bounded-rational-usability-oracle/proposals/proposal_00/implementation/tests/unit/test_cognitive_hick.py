"""Unit tests for usability_oracle.cognitive.hick.HickHymanLaw.

Tests cover choice reaction time prediction for equiprobable and
unequal-probability stimuli, Shannon entropy, information gain (KL
divergence), effective alternatives, practice effects, stimulus-response
compatibility, interval-valued prediction, monotonicity, and edge cases.

References
----------
Hick, W. E. (1952). On the rate of gain of information. *QJEP*, 4(1), 11-26.
Hyman, R. (1953). Stimulus information as a determinant of reaction time.
    *J Exp Psychol*, 45(3), 188-196.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.cognitive.hick import HickHymanLaw
from usability_oracle.interval.interval import Interval


# ------------------------------------------------------------------ #
# Default constants
# ------------------------------------------------------------------ #


class TestHickDefaults:
    """Verify published default parameter values."""

    def test_default_a(self) -> None:
        """DEFAULT_A should be 0.200 s (Hick, 1952)."""
        assert HickHymanLaw.DEFAULT_A == pytest.approx(0.200)

    def test_default_b(self) -> None:
        """DEFAULT_B should be 0.155 s/bit (Hick, 1952)."""
        assert HickHymanLaw.DEFAULT_B == pytest.approx(0.155)


# ------------------------------------------------------------------ #
# Core prediction — equiprobable alternatives
# ------------------------------------------------------------------ #


class TestHickPredict:
    """Tests for HickHymanLaw.predict() — equiprobable stimuli."""

    def test_basic_prediction(self) -> None:
        """predict(n) = a + b * log2(n).

        With n=4, defaults → 0.200 + 0.155 * log2(4) = 0.200 + 0.310 = 0.510.
        """
        expected = 0.200 + 0.155 * math.log2(4)
        result = HickHymanLaw.predict(4)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_custom_a_b(self) -> None:
        """predict() should accept custom a and b parameters.

        a=0.1, b=0.2, n=8 → 0.1 + 0.2 * log2(8) = 0.1 + 0.6 = 0.7.
        """
        expected = 0.1 + 0.2 * math.log2(8)
        result = HickHymanLaw.predict(8, a=0.1, b=0.2)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_single_alternative(self) -> None:
        """With n=1, RT should equal a (simple reaction time).

        log2(1) = 0, so the information component vanishes.
        """
        result = HickHymanLaw.predict(1)
        assert result == pytest.approx(HickHymanLaw.DEFAULT_A, rel=1e-9)

    def test_raises_on_zero_alternatives(self) -> None:
        """predict() must raise ValueError when n < 1."""
        with pytest.raises(ValueError, match="n_alternatives must be >= 1"):
            HickHymanLaw.predict(0)

    def test_returns_float(self) -> None:
        """predict() should always return a Python float."""
        assert isinstance(HickHymanLaw.predict(5), float)


# ------------------------------------------------------------------ #
# Unequal probabilities (Hyman, 1953)
# ------------------------------------------------------------------ #


class TestPredictUnequalProbabilities:
    """Tests for HickHymanLaw.predict_unequal_probabilities()."""

    def test_uniform_matches_predict(self) -> None:
        """Uniform probabilities for n items should match predict(n).

        For n=4 uniform: H = log2(4) = 2 bits, so RT should match.
        """
        probs = [0.25, 0.25, 0.25, 0.25]
        result = HickHymanLaw.predict_unequal_probabilities(probs)
        expected = HickHymanLaw.predict(4)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_concentrated_probability(self) -> None:
        """A very skewed distribution should yield lower RT.

        If one alternative dominates, entropy is low → fast RT.
        """
        concentrated = [0.97, 0.01, 0.01, 0.01]
        uniform_rt = HickHymanLaw.predict(4)
        result = HickHymanLaw.predict_unequal_probabilities(concentrated)
        assert result < uniform_rt

    def test_raises_on_bad_sum(self) -> None:
        """Must raise ValueError if probabilities don't sum to ~1."""
        with pytest.raises(ValueError, match="Probabilities must sum to 1"):
            HickHymanLaw.predict_unequal_probabilities([0.5, 0.3])

    def test_raises_on_negative_probs(self) -> None:
        """Must raise ValueError for negative probabilities."""
        with pytest.raises(ValueError, match="non-negative"):
            HickHymanLaw.predict_unequal_probabilities([1.2, -0.2])


# ------------------------------------------------------------------ #
# Shannon entropy
# ------------------------------------------------------------------ #


class TestEntropy:
    """Tests for HickHymanLaw.entropy()."""

    def test_uniform_entropy(self) -> None:
        """Uniform distribution over n items → H = log2(n).

        H([0.25, 0.25, 0.25, 0.25]) = log2(4) = 2.0 bits.
        """
        result = HickHymanLaw.entropy([0.25, 0.25, 0.25, 0.25])
        assert result == pytest.approx(2.0, rel=1e-9)

    def test_certain_entropy(self) -> None:
        """Certain outcome (p=1) → H = 0.

        A degenerate distribution with all mass on one outcome has zero
        information content.
        """
        result = HickHymanLaw.entropy([1.0, 0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_equal_probs_maximum_entropy(self) -> None:
        """Equal probabilities should yield the maximum possible entropy.

        For any n, uniform is the max-entropy distribution, with H = log2(n).
        """
        n = 8
        probs = [1.0 / n] * n
        h = HickHymanLaw.entropy(probs)
        assert h == pytest.approx(math.log2(n), rel=1e-9)

    def test_entropy_non_negative(self) -> None:
        """Shannon entropy is always >= 0."""
        assert HickHymanLaw.entropy([0.7, 0.2, 0.1]) >= 0.0


# ------------------------------------------------------------------ #
# Information gain (KL divergence)
# ------------------------------------------------------------------ #


class TestInformationGain:
    """Tests for HickHymanLaw.information_gain()."""

    def test_same_distribution_zero_divergence(self) -> None:
        """KL(P || P) = 0 for any P."""
        p = [0.25, 0.25, 0.25, 0.25]
        result = HickHymanLaw.information_gain(p, p)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_kl_nonnegative(self) -> None:
        """KL divergence is always non-negative (Gibbs' inequality)."""
        prior = [0.25, 0.25, 0.25, 0.25]
        posterior = [0.5, 0.3, 0.1, 0.1]
        result = HickHymanLaw.information_gain(prior, posterior)
        assert result >= -1e-12

    def test_raises_on_shape_mismatch(self) -> None:
        """Must raise ValueError for distributions of different lengths."""
        with pytest.raises(ValueError, match="same length"):
            HickHymanLaw.information_gain([0.5, 0.5], [0.33, 0.33, 0.34])


# ------------------------------------------------------------------ #
# Effective alternatives
# ------------------------------------------------------------------ #


class TestEffectiveAlternatives:
    """Tests for HickHymanLaw.effective_alternatives()."""

    def test_uniform_returns_n(self) -> None:
        """For uniform distribution over n items, effective_alternatives = n."""
        probs = [0.25, 0.25, 0.25, 0.25]
        result = HickHymanLaw.effective_alternatives(probs)
        assert result == pytest.approx(4.0, rel=1e-6)

    def test_concentrated_returns_less(self) -> None:
        """Skewed distribution → fewer effective alternatives than n."""
        probs = [0.9, 0.05, 0.05]
        result = HickHymanLaw.effective_alternatives(probs)
        assert result < 3.0


# ------------------------------------------------------------------ #
# Practice effects
# ------------------------------------------------------------------ #


class TestPracticeEffects:
    """Tests for practice_factor() and predict_with_practice()."""

    def test_practice_factor_first_trial(self) -> None:
        """First trial (trials=1) → factor = 1.0 (no speed-up)."""
        result = HickHymanLaw.practice_factor(1)
        assert result == pytest.approx(1.0, rel=1e-9)

    def test_practice_factor_decreases(self) -> None:
        """More practice trials → smaller factor (faster response)."""
        f10 = HickHymanLaw.practice_factor(10)
        f100 = HickHymanLaw.practice_factor(100)
        assert f100 < f10 < 1.0

    def test_practice_factor_raises_on_zero(self) -> None:
        """practice_factor() must raise ValueError for trials < 1."""
        with pytest.raises(ValueError, match="trials must be >= 1"):
            HickHymanLaw.practice_factor(0)

    def test_predict_with_practice_faster(self) -> None:
        """Practiced RT should be less than or equal to unpracticed RT."""
        base = HickHymanLaw.predict(8)
        practiced = HickHymanLaw.predict_with_practice(8, trials=50)
        assert practiced < base


# ------------------------------------------------------------------ #
# Stimulus-response compatibility
# ------------------------------------------------------------------ #


class TestStimulusResponseCompatibility:
    """Tests for HickHymanLaw.stimulus_response_compatibility()."""

    def test_default_compatibility(self) -> None:
        """compatibility_factor=1.0 → same as base predict().

        The default compatibility is neutral.
        """
        result = HickHymanLaw.stimulus_response_compatibility(4, 1.0)
        expected = HickHymanLaw.predict(4)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_high_compatibility_faster(self) -> None:
        """High compatibility (c > 1) should reduce RT.

        A spatially compatible mapping reduces the effective slope.
        """
        compatible = HickHymanLaw.stimulus_response_compatibility(8, 2.0)
        neutral = HickHymanLaw.stimulus_response_compatibility(8, 1.0)
        assert compatible < neutral

    def test_raises_on_zero_factor(self) -> None:
        """Must raise ValueError for compatibility_factor <= 0."""
        with pytest.raises(ValueError, match="compatibility_factor must be > 0"):
            HickHymanLaw.stimulus_response_compatibility(4, 0.0)


# ------------------------------------------------------------------ #
# Interval-valued prediction
# ------------------------------------------------------------------ #


class TestPredictInterval:
    """Tests for HickHymanLaw.predict_interval() with Interval inputs."""

    def test_point_intervals_match_scalar(self) -> None:
        """Degenerate (point) intervals should match the scalar predict().

        This verifies consistency between the point and interval APIs.
        """
        n = 6
        point = HickHymanLaw.predict(n)
        ivl = HickHymanLaw.predict_interval(
            n,
            a=Interval.from_value(HickHymanLaw.DEFAULT_A),
            b=Interval.from_value(HickHymanLaw.DEFAULT_B),
        )
        assert ivl.low == pytest.approx(point, rel=1e-6)
        assert ivl.high == pytest.approx(point, rel=1e-6)

    def test_single_alternative_returns_a(self) -> None:
        """With n=1, interval prediction should return the a interval."""
        a_ivl = Interval(0.18, 0.22)
        b_ivl = Interval(0.14, 0.16)
        result = HickHymanLaw.predict_interval(1, a_ivl, b_ivl)
        assert result.low == pytest.approx(a_ivl.low)
        assert result.high == pytest.approx(a_ivl.high)

    def test_interval_widens_with_uncertainty(self) -> None:
        """Wider parameter intervals → wider output interval."""
        narrow = HickHymanLaw.predict_interval(
            8, Interval(0.199, 0.201), Interval(0.154, 0.156)
        )
        wide = HickHymanLaw.predict_interval(
            8, Interval(0.15, 0.25), Interval(0.10, 0.20)
        )
        assert wide.width > narrow.width


# ------------------------------------------------------------------ #
# Monotonicity
# ------------------------------------------------------------------ #


class TestMonotonicity:
    """Monotonicity properties of the Hick-Hyman law."""

    def test_more_alternatives_more_time(self) -> None:
        """RT must increase monotonically with the number of alternatives.

        For any n1 < n2, predict(n1) < predict(n2).
        """
        times = [HickHymanLaw.predict(n) for n in [1, 2, 4, 8, 16, 32]]
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1], (
                f"RT should increase: n={2**i} → {times[i]}, "
                f"n={2**(i+1)} → {times[i+1]}"
            )
