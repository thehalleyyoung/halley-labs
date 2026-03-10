"""Unit tests for usability_oracle.differential.mechanisms — Privacy mechanisms.

Tests cover Laplace mechanism ε-DP guarantee, Gaussian mechanism, exponential
mechanism, sensitivity computation, and noise calibration.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from usability_oracle.differential.types import (
    MechanismType,
    NoiseConfig,
    PrivacyBudget,
)
from usability_oracle.differential.mechanisms import (
    above_threshold,
    exponential_mechanism,
    gaussian_mechanism,
    gaussian_mechanism_vector,
    gaussian_scale,
    geometric_mechanism,
    laplace_mechanism,
    laplace_mechanism_vector,
    laplace_scale,
    noise_config_gaussian,
    noise_config_laplace,
    randomized_response,
    report_noisy_max,
    sensitivity_count,
    sensitivity_mean,
    sensitivity_sum,
    sparse_vector_technique,
    truncated_laplace,
)


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity Computation
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivity:
    """Test sensitivity function computation."""

    def test_count_sensitivity(self):
        assert sensitivity_count() == 1.0

    def test_sum_sensitivity(self):
        assert sensitivity_sum(clipping_bound=5.0) == 5.0

    def test_mean_sensitivity(self):
        s = sensitivity_mean(clipping_bound=10.0, n=100)
        assert s == pytest.approx(0.1)

    def test_sum_sensitivity_with_zero_clip(self):
        assert sensitivity_sum(clipping_bound=0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Laplace Mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestLaplaceMechanism:
    """Test Laplace mechanism and ε-DP guarantee."""

    def test_laplace_scale_computation(self):
        scale = laplace_scale(sensitivity=1.0, epsilon=1.0)
        assert scale == pytest.approx(1.0)

    def test_laplace_scale_inversely_proportional_to_epsilon(self):
        s1 = laplace_scale(sensitivity=1.0, epsilon=1.0)
        s2 = laplace_scale(sensitivity=1.0, epsilon=2.0)
        assert s1 == pytest.approx(2 * s2)

    def test_laplace_mechanism_preserves_expectation(self):
        """Mean of many Laplace noise samples should be close to the true value."""
        rng = np.random.default_rng(42)
        true_value = 5.0
        samples = [
            laplace_mechanism(true_value, sensitivity=1.0, epsilon=1.0, rng=rng)
            for _ in range(10000)
        ]
        assert abs(np.mean(samples) - true_value) < 0.1

    def test_laplace_mechanism_noise_scale(self):
        """Variance of Laplace(b) is 2b², so std ≈ b√2."""
        rng = np.random.default_rng(42)
        eps = 1.0
        sens = 1.0
        expected_b = sens / eps
        samples = [
            laplace_mechanism(0.0, sensitivity=sens, epsilon=eps, rng=rng)
            for _ in range(10000)
        ]
        empirical_std = np.std(samples)
        expected_std = expected_b * math.sqrt(2)
        assert abs(empirical_std - expected_std) < 0.1

    def test_laplace_vector(self):
        rng = np.random.default_rng(42)
        values = np.array([1.0, 2.0, 3.0])
        noisy = laplace_mechanism_vector(values, sensitivity=1.0, epsilon=1.0, rng=rng)
        assert noisy.shape == (3,)
        # Should be close to original but not identical
        assert not np.array_equal(values, noisy)

    def test_laplace_dp_guarantee_empirical(self):
        """Empirical check: ratio of P(M(x)∈S)/P(M(x')∈S) ≈ exp(ε) for adjacent inputs."""
        rng = np.random.default_rng(42)
        eps = 1.0
        n_samples = 50000
        samples_a = np.array([
            laplace_mechanism(0.0, 1.0, eps, rng=rng) for _ in range(n_samples)
        ])
        samples_b = np.array([
            laplace_mechanism(1.0, 1.0, eps, rng=rng) for _ in range(n_samples)
        ])
        # Check a specific region: [-0.5, 0.5]
        count_a = np.sum((samples_a >= -0.5) & (samples_a <= 0.5))
        count_b = np.sum((samples_b >= -0.5) & (samples_b <= 0.5))
        if count_b > 0:
            ratio = count_a / count_b
            assert ratio <= math.exp(eps) + 0.5  # allow some slack

    def test_truncated_laplace(self):
        rng = np.random.default_rng(42)
        result = truncated_laplace(5.0, 1.0, 1.0, lower=0.0, upper=10.0, rng=rng)
        assert 0.0 <= result <= 10.0


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian Mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestGaussianMechanism:
    """Test Gaussian mechanism for (ε,δ)-DP."""

    def test_gaussian_scale_computation(self):
        scale = gaussian_scale(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert scale > 0

    def test_gaussian_scale_increases_with_delta_decrease(self):
        s1 = gaussian_scale(sensitivity=1.0, epsilon=1.0, delta=1e-3)
        s2 = gaussian_scale(sensitivity=1.0, epsilon=1.0, delta=1e-6)
        assert s2 > s1

    def test_gaussian_mechanism_preserves_expectation(self):
        rng = np.random.default_rng(42)
        true_value = 3.0
        samples = [
            gaussian_mechanism(true_value, 1.0, 1.0, 1e-5, rng=rng) for _ in range(10000)
        ]
        assert abs(np.mean(samples) - true_value) < 0.15

    def test_gaussian_vector(self):
        rng = np.random.default_rng(42)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        noisy = gaussian_mechanism_vector(values, 1.0, 1.0, 1e-5, rng=rng)
        assert noisy.shape == (4,)

    def test_noise_config_gaussian(self):
        config = noise_config_gaussian(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert isinstance(config, NoiseConfig)
        assert config.mechanism == MechanismType.GAUSSIAN
        assert config.scale > 0


# ═══════════════════════════════════════════════════════════════════════════
# Exponential Mechanism
# ═══════════════════════════════════════════════════════════════════════════


class TestExponentialMechanism:
    """Test exponential mechanism for categorical selection."""

    def test_selects_highest_score_with_high_epsilon(self):
        candidates = ["a", "b", "c"]
        scores = {"a": 1.0, "b": 5.0, "c": 2.0}
        rng = np.random.default_rng(42)
        counts = Counter()
        for _ in range(1000):
            selected = exponential_mechanism(
                candidates,
                score_fn=lambda x: scores[x],
                sensitivity=1.0,
                epsilon=10.0,
                rng=rng,
            )
            counts[selected] += 1
        # With high epsilon, "b" should be selected most often
        assert counts["b"] > counts["a"]
        assert counts["b"] > counts["c"]

    def test_low_epsilon_more_uniform(self):
        candidates = ["a", "b", "c"]
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        rng = np.random.default_rng(42)
        counts = Counter()
        for _ in range(3000):
            selected = exponential_mechanism(
                candidates,
                score_fn=lambda x: scores[x],
                sensitivity=1.0,
                epsilon=0.01,
                rng=rng,
            )
            counts[selected] += 1
        # Should be relatively uniform
        fractions = [counts[c] / 3000 for c in candidates]
        assert max(fractions) - min(fractions) < 0.3

    def test_single_candidate(self):
        rng = np.random.default_rng(42)
        result = exponential_mechanism(
            ["only"],
            score_fn=lambda x: 1.0,
            sensitivity=1.0,
            epsilon=1.0,
            rng=rng,
        )
        assert result == "only"


# ═══════════════════════════════════════════════════════════════════════════
# Other Mechanisms
# ═══════════════════════════════════════════════════════════════════════════


class TestOtherMechanisms:
    """Test randomized response, report-noisy-max, geometric, SVT."""

    def test_randomized_response_preserves_rate(self):
        rng = np.random.default_rng(42)
        n = 10000
        true_rate = 0.7
        responses = [
            randomized_response(rng.random() < true_rate, epsilon=2.0, rng=rng)
            for _ in range(n)
        ]
        observed_rate = sum(responses) / n
        # Should be closer to true rate with higher epsilon
        assert abs(observed_rate - true_rate) < 0.1

    def test_report_noisy_max(self):
        rng = np.random.default_rng(42)
        scores = [1.0, 5.0, 3.0]
        counts = Counter()
        for _ in range(1000):
            idx = report_noisy_max(scores, 1.0, 5.0, rng=rng)
            counts[idx] += 1
        # Index 1 (score 5.0) should be selected most often
        assert counts[1] > counts[0]

    def test_geometric_mechanism(self):
        rng = np.random.default_rng(42)
        results = [geometric_mechanism(10, 1, 1.0, rng=rng) for _ in range(1000)]
        assert abs(np.mean(results) - 10) < 1.0

    def test_sparse_vector_technique(self):
        rng = np.random.default_rng(42)
        queries = [0.5, 1.5, 0.3, 2.0, 0.1]
        results = sparse_vector_technique(
            queries, threshold=1.0, sensitivity=1.0, epsilon=2.0,
            max_above=2, rng=rng,
        )
        assert isinstance(results, list)
        assert len(results) == len(queries)

    def test_above_threshold(self):
        rng = np.random.default_rng(42)
        queries = [0.1, 0.2, 5.0, 0.3]
        idx = above_threshold(queries, threshold=1.0, sensitivity=1.0, epsilon=5.0, rng=rng)
        assert isinstance(idx, int)
        assert 0 <= idx < len(queries)

    def test_noise_config_laplace(self):
        config = noise_config_laplace(sensitivity=1.0, epsilon=1.0)
        assert isinstance(config, NoiseConfig)
        assert config.mechanism == MechanismType.LAPLACE
