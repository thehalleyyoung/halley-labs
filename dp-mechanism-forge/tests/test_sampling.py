"""Comprehensive tests for dp_forge.sampling module.

Tests cover:
  - AliasMethodSampler: build, sample, sample_batch, log_probability, entropy
  - InverseCDFSampler: build, sample, sample_batch, quantile, median
  - RejectionSampler: build, sample, sample_batch, acceptance_rate
  - SecureRNG: random, integers, random_array
  - constant_time_sample, sample_with_seed
  - DiscreteDistribution: pmf, log_pmf, cdf, sample, mean, variance, entropy, mode
  - MixtureDistribution: sample, mean, variance, pmf
  - TruncatedDistribution: sample, mean, pmf
  - ConditionalDistribution: sample, mean, conditioning_probability
  - MechanismSampler: sample, sample_values, estimate_mse, estimate_mae
  - Statistical tests: chi_squared_test, ks_test, uniformity_test
  - build_sampler factory
  - Numerical utilities: _logsumexp, _log_subtract, _normalize_probabilities
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from dp_forge.sampling import (
    AliasMethodSampler,
    InverseCDFSampler,
    RejectionSampler,
    SecureRNG,
    DiscreteDistribution,
    MixtureDistribution,
    TruncatedDistribution,
    ConditionalDistribution,
    MechanismSampler,
    chi_squared_test,
    ks_test,
    uniformity_test,
    build_sampler,
    constant_time_sample,
    sample_with_seed,
    _logsumexp,
    _log_subtract,
    _normalize_probabilities,
    _validate_probabilities,
)
from dp_forge.types import (
    ExtractedMechanism,
    QuerySpec,
    SamplingConfig,
    SamplingMethod,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_probs():
    """Uniform distribution over 4 outcomes."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def skewed_probs():
    """Skewed distribution over 4 outcomes."""
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def simple_mechanism():
    """A small 3×5 mechanism for testing MechanismSampler."""
    rng = np.random.default_rng(42)
    p = rng.dirichlet(np.ones(5), size=3)
    return ExtractedMechanism(p_final=p)


@pytest.fixture
def simple_spec():
    """A simple QuerySpec for testing."""
    return QuerySpec(
        query_values=np.array([0.0, 1.0, 2.0]),
        domain="test",
        sensitivity=1.0,
        epsilon=1.0,
        delta=0.0,
        k=5,
    )


@pytest.fixture
def deterministic_rng():
    """A seeded numpy RNG for reproducibility."""
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

class TestLogsumexp:
    """Tests for _logsumexp."""

    def test_single_element(self):
        assert _logsumexp(np.array([2.0])) == pytest.approx(2.0)

    def test_two_equal_elements(self):
        result = _logsumexp(np.array([0.0, 0.0]))
        assert result == pytest.approx(math.log(2.0))

    def test_large_values_no_overflow(self):
        vals = np.array([1000.0, 1000.0, 1000.0])
        result = _logsumexp(vals)
        assert result == pytest.approx(1000.0 + math.log(3.0))

    def test_very_negative_values_no_underflow(self):
        vals = np.array([-1000.0, -1001.0])
        result = _logsumexp(vals)
        expected = -1000.0 + math.log(1.0 + math.exp(-1.0))
        assert result == pytest.approx(expected)

    def test_empty_array(self):
        assert _logsumexp(np.array([])) == -np.inf

    def test_negative_inf(self):
        result = _logsumexp(np.array([-np.inf, 0.0]))
        assert result == pytest.approx(0.0)


class TestLogSubtract:
    """Tests for _log_subtract."""

    def test_basic(self):
        result = _log_subtract(math.log(3.0), math.log(1.0))
        assert result == pytest.approx(math.log(2.0))

    def test_equal_values_returns_neg_inf(self):
        assert _log_subtract(5.0, 5.0) == -np.inf

    def test_log_b_neg_inf(self):
        assert _log_subtract(3.0, -np.inf) == pytest.approx(3.0)

    def test_raises_when_log_a_less_than_log_b(self):
        with pytest.raises(ValueError, match="log_subtract requires"):
            _log_subtract(1.0, 5.0)


class TestNormalizeProbabilities:
    """Tests for _normalize_probabilities."""

    def test_already_normalized(self):
        p = np.array([0.5, 0.5])
        result = _normalize_probabilities(p)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_rescales(self):
        p = np.array([1.0, 1.0])
        result = _normalize_probabilities(p)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_clamps_negatives(self):
        p = np.array([-0.001, 0.5, 0.501])
        result = _normalize_probabilities(p)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result.sum() == pytest.approx(1.0)

    def test_raises_on_all_zero(self):
        with pytest.raises(ValueError, match="sums to zero"):
            _normalize_probabilities(np.array([0.0, 0.0]))


class TestValidateProbabilities:
    """Tests for _validate_probabilities."""

    def test_valid_1d(self):
        result = _validate_probabilities(np.array([0.3, 0.7]))
        assert result.sum() == pytest.approx(1.0)

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            _validate_probabilities(np.array([[0.5, 0.5]]))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            _validate_probabilities(np.array([]))

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="contains NaN"):
            _validate_probabilities(np.array([0.5, np.nan]))


# ---------------------------------------------------------------------------
# AliasMethodSampler
# ---------------------------------------------------------------------------

class TestAliasMethodSampler:
    """Tests for AliasMethodSampler."""

    def test_build_returns_self(self, uniform_probs):
        sampler = AliasMethodSampler()
        result = sampler.build(uniform_probs)
        assert result is sampler

    def test_sample_before_build_raises(self):
        sampler = AliasMethodSampler()
        with pytest.raises(RuntimeError, match="build.*must be called"):
            sampler.sample()

    def test_k_set_after_build(self, skewed_probs):
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        assert sampler.k == 4

    def test_sample_returns_valid_range(self, skewed_probs, deterministic_rng):
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        for _ in range(100):
            s = sampler.sample(rng=deterministic_rng)
            assert 0 <= s < 4

    def test_sample_batch_shape(self, uniform_probs, deterministic_rng):
        sampler = AliasMethodSampler()
        sampler.build(uniform_probs)
        batch = sampler.sample_batch(1000, rng=deterministic_rng)
        assert batch.shape == (1000,)
        assert batch.dtype == np.int64

    def test_sample_batch_zero(self, uniform_probs):
        sampler = AliasMethodSampler()
        sampler.build(uniform_probs)
        batch = sampler.sample_batch(0)
        assert len(batch) == 0

    def test_sample_batch_negative_raises(self, uniform_probs):
        sampler = AliasMethodSampler()
        sampler.build(uniform_probs)
        with pytest.raises(ValueError, match="n must be >= 0"):
            sampler.sample_batch(-1)

    def test_uniform_distribution_chi_squared(self, uniform_probs):
        """Alias samples from a uniform distribution should pass chi-squared test."""
        sampler = AliasMethodSampler()
        sampler.build(uniform_probs)
        rng = np.random.default_rng(99)
        samples = sampler.sample_batch(50_000, rng=rng)
        result = chi_squared_test(samples, uniform_probs)
        assert result["passed"], f"Chi-squared failed: p={result['p_value']:.4f}"

    def test_skewed_distribution_frequencies(self, skewed_probs):
        """Sample frequencies should approximate the target distribution."""
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        rng = np.random.default_rng(42)
        n = 100_000
        samples = sampler.sample_batch(n, rng=rng)
        counts = np.bincount(samples, minlength=4) / n
        np.testing.assert_allclose(counts, skewed_probs, atol=0.01)

    def test_log_probability(self, skewed_probs):
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        for i, p in enumerate(skewed_probs):
            assert sampler.log_probability(i) == pytest.approx(math.log(p))

    def test_log_probability_out_of_range(self, uniform_probs):
        sampler = AliasMethodSampler()
        sampler.build(uniform_probs)
        with pytest.raises(IndexError, match="out of range"):
            sampler.log_probability(10)
        with pytest.raises(IndexError, match="out of range"):
            sampler.log_probability(-1)

    def test_entropy_uniform(self, uniform_probs):
        sampler = AliasMethodSampler()
        sampler.build(uniform_probs)
        expected = -4 * 0.25 * math.log(0.25)
        assert sampler.entropy == pytest.approx(expected)

    def test_entropy_degenerate(self):
        """Degenerate distribution (all mass on one outcome) has zero entropy."""
        p = np.array([1.0, 0.0, 0.0])
        sampler = AliasMethodSampler()
        sampler.build(p)
        assert sampler.entropy == pytest.approx(0.0)

    def test_original_probabilities_returns_copy(self, skewed_probs):
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        orig = sampler.original_probabilities
        np.testing.assert_allclose(orig, skewed_probs)
        orig[0] = 999.0  # mutating the copy should not affect sampler
        np.testing.assert_allclose(sampler.original_probabilities, skewed_probs)

    def test_sample_conditional(self, skewed_probs):
        """Conditional sampling should only return outcomes satisfying the condition."""
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        rng = np.random.default_rng(7)
        for _ in range(50):
            s = sampler.sample_conditional(lambda x: x >= 2, rng=rng)
            assert s >= 2

    def test_sample_conditional_impossible(self):
        """Conditioning on impossible event should raise."""
        p = np.array([1.0, 0.0])
        sampler = AliasMethodSampler()
        sampler.build(p)
        with pytest.raises(RuntimeError, match="Conditional sampling failed"):
            sampler.sample_conditional(
                lambda x: x == 1, max_attempts=100
            )

    def test_repr(self, uniform_probs):
        sampler = AliasMethodSampler()
        assert "not built" in repr(sampler)
        sampler.build(uniform_probs)
        assert "k=4" in repr(sampler)

    def test_build_with_large_distribution(self):
        """Build and sample from a large distribution (k=1000)."""
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(1000))
        sampler = AliasMethodSampler()
        sampler.build(p)
        samples = sampler.sample_batch(10_000, rng=rng)
        assert samples.min() >= 0
        assert samples.max() < 1000

    def test_build_rejects_invalid(self):
        with pytest.raises(ValueError):
            AliasMethodSampler().build(np.array([]))

    def test_degenerate_single_outcome(self):
        """Single-outcome distribution always returns 0."""
        sampler = AliasMethodSampler()
        sampler.build(np.array([1.0]))
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert sampler.sample(rng=rng) == 0

    def test_reproducibility(self, skewed_probs):
        """Same seed should produce identical samples."""
        sampler = AliasMethodSampler()
        sampler.build(skewed_probs)
        s1 = sampler.sample_batch(100, rng=np.random.default_rng(42))
        s2 = sampler.sample_batch(100, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# InverseCDFSampler
# ---------------------------------------------------------------------------

class TestInverseCDFSampler:
    """Tests for InverseCDFSampler."""

    def test_build_returns_self(self, uniform_probs):
        sampler = InverseCDFSampler()
        result = sampler.build(uniform_probs)
        assert result is sampler

    def test_sample_before_build_raises(self):
        sampler = InverseCDFSampler()
        with pytest.raises(RuntimeError, match="build.*must be called"):
            sampler.sample()

    def test_cdf_monotonic(self, skewed_probs):
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        for i in range(1, sampler.k):
            assert sampler.cdf[i] >= sampler.cdf[i - 1]
        assert sampler.cdf[-1] == pytest.approx(1.0)

    def test_sample_returns_valid_range(self, skewed_probs, deterministic_rng):
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        for _ in range(100):
            s = sampler.sample(rng=deterministic_rng)
            assert 0 <= s < 4

    def test_sample_batch_shape(self, uniform_probs, deterministic_rng):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        batch = sampler.sample_batch(500, rng=deterministic_rng)
        assert batch.shape == (500,)
        assert batch.dtype == np.int64

    def test_sample_batch_zero(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        batch = sampler.sample_batch(0)
        assert len(batch) == 0

    def test_sample_batch_negative_raises(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        with pytest.raises(ValueError, match="n must be >= 0"):
            sampler.sample_batch(-1)

    def test_skewed_distribution_frequencies(self, skewed_probs):
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        rng = np.random.default_rng(99)
        n = 100_000
        samples = sampler.sample_batch(n, rng=rng)
        counts = np.bincount(samples, minlength=4) / n
        np.testing.assert_allclose(counts, skewed_probs, atol=0.01)

    def test_quantile_uniform(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        assert sampler.quantile(0.0) == 0
        assert sampler.quantile(0.24) == 0
        assert sampler.quantile(0.26) == 1
        assert sampler.quantile(1.0) == 3

    def test_quantile_invalid_raises(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        with pytest.raises(ValueError, match="must be in"):
            sampler.quantile(-0.1)
        with pytest.raises(ValueError, match="must be in"):
            sampler.quantile(1.1)

    def test_median_property(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        assert sampler.median == sampler.quantile(0.5)

    def test_quantile_value_with_grid(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        grid = np.array([10.0, 20.0, 30.0, 40.0])
        sampler = InverseCDFSampler()
        sampler.build(probs, grid=grid)
        assert sampler.quantile_value(0.5) == 20.0

    def test_quantile_value_no_grid_raises(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        with pytest.raises(RuntimeError, match="No grid"):
            sampler.quantile_value(0.5)

    def test_log_probability(self, skewed_probs):
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        for i, p in enumerate(skewed_probs):
            assert sampler.log_probability(i) == pytest.approx(math.log(p))

    def test_log_probability_out_of_range(self, uniform_probs):
        sampler = InverseCDFSampler()
        sampler.build(uniform_probs)
        with pytest.raises(IndexError, match="out of range"):
            sampler.log_probability(5)

    def test_original_probabilities(self, skewed_probs):
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        np.testing.assert_allclose(sampler.original_probabilities, skewed_probs)

    def test_repr(self, uniform_probs):
        sampler = InverseCDFSampler()
        assert "not built" in repr(sampler)
        sampler.build(uniform_probs)
        assert "k=4" in repr(sampler)

    def test_reproducibility(self, skewed_probs):
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        s1 = sampler.sample_batch(100, rng=np.random.default_rng(7))
        s2 = sampler.sample_batch(100, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(s1, s2)

    def test_chi_squared_match(self, skewed_probs):
        """InverseCDF samples should pass chi-squared test."""
        sampler = InverseCDFSampler()
        sampler.build(skewed_probs)
        rng = np.random.default_rng(55)
        samples = sampler.sample_batch(50_000, rng=rng)
        result = chi_squared_test(samples, skewed_probs)
        assert result["passed"], f"Chi-squared failed: p={result['p_value']:.4f}"


# ---------------------------------------------------------------------------
# RejectionSampler
# ---------------------------------------------------------------------------

class TestRejectionSampler:
    """Tests for RejectionSampler."""

    def _build_bernoulli_sampler(self, p_target: float = 0.6):
        """Build a rejection sampler for a Bernoulli(p_target) distribution."""
        target_pdf = lambda x: p_target if x == 1 else (1.0 - p_target) if x == 0 else 0.0
        proposal_sample = lambda rng: rng.integers(0, 2)
        proposal_pdf = lambda x: 0.5
        M = max(p_target, 1.0 - p_target) / 0.5
        sampler = RejectionSampler()
        sampler.build(target_pdf, proposal_sample, proposal_pdf, M=M)
        return sampler

    def test_build_returns_self(self):
        sampler = RejectionSampler()
        result = sampler.build(
            lambda x: 1.0, lambda rng: 0, lambda x: 1.0, M=1.0
        )
        assert result is sampler

    def test_sample_before_build_raises(self):
        sampler = RejectionSampler()
        with pytest.raises(RuntimeError, match="build.*must be called"):
            sampler.sample()

    def test_build_negative_M_raises(self):
        sampler = RejectionSampler()
        with pytest.raises(ValueError, match="M must be > 0"):
            sampler.build(lambda x: 1.0, lambda rng: 0, lambda x: 1.0, M=-1.0)

    def test_sample_returns_valid(self):
        sampler = self._build_bernoulli_sampler()
        rng = np.random.default_rng(42)
        for _ in range(50):
            s = sampler.sample(rng=rng)
            assert s in (0, 1)

    def test_sample_batch(self):
        sampler = self._build_bernoulli_sampler()
        rng = np.random.default_rng(42)
        batch = sampler.sample_batch(1000, rng=rng)
        assert len(batch) == 1000
        assert all(s in (0, 1) for s in batch)

    def test_distribution_accuracy(self):
        """Rejection samples should match the target distribution."""
        p_target = 0.7
        sampler = self._build_bernoulli_sampler(p_target)
        rng = np.random.default_rng(123)
        samples = sampler.sample_batch(20_000, rng=rng)
        freq_1 = sum(1 for s in samples if s == 1) / len(samples)
        assert freq_1 == pytest.approx(p_target, abs=0.02)

    def test_acceptance_rate_positive(self):
        sampler = self._build_bernoulli_sampler()
        rng = np.random.default_rng(0)
        sampler.sample_batch(500, rng=rng)
        rate = sampler.acceptance_rate()
        assert 0.0 < rate <= 1.0

    def test_acceptance_rate_before_sampling(self):
        sampler = self._build_bernoulli_sampler()
        assert sampler.acceptance_rate() == 0.0

    def test_theoretical_acceptance_rate(self):
        sampler = RejectionSampler()
        sampler.build(
            lambda x: 1.0, lambda rng: 0, lambda x: 1.0, M=2.0
        )
        assert sampler.theoretical_acceptance_rate() == pytest.approx(0.5)

    def test_reset_counters(self):
        sampler = self._build_bernoulli_sampler()
        rng = np.random.default_rng(0)
        sampler.sample_batch(100, rng=rng)
        assert sampler.acceptance_rate() > 0
        sampler.reset_counters()
        assert sampler.acceptance_rate() == 0.0

    def test_repr(self):
        sampler = RejectionSampler()
        assert "not built" in repr(sampler)
        sampler.build(
            lambda x: 1.0, lambda rng: 0, lambda x: 1.0, M=2.0
        )
        rng = np.random.default_rng(0)
        sampler.sample(rng=rng)
        assert "M=2.00" in repr(sampler)

    def test_discrete_target_frequencies(self):
        """Rejection sampler on a 3-outcome discrete target."""
        target_probs = [0.2, 0.5, 0.3]
        target_pdf = lambda x: target_probs[x] if 0 <= x <= 2 else 0.0
        proposal_sample = lambda rng: rng.integers(0, 3)
        proposal_pdf = lambda x: 1.0 / 3.0
        M = max(target_probs) / (1.0 / 3.0)
        sampler = RejectionSampler()
        sampler.build(target_pdf, proposal_sample, proposal_pdf, M=M)
        rng = np.random.default_rng(77)
        samples = sampler.sample_batch(30_000, rng=rng)
        counts = np.bincount(samples, minlength=3) / len(samples)
        np.testing.assert_allclose(counts, target_probs, atol=0.02)


# ---------------------------------------------------------------------------
# SecureRNG
# ---------------------------------------------------------------------------

class TestSecureRNG:
    """Tests for SecureRNG."""

    def test_random_in_range(self):
        rng = SecureRNG()
        for _ in range(100):
            v = rng.random()
            assert 0.0 <= v < 1.0

    def test_integers_in_range(self):
        rng = SecureRNG()
        for _ in range(100):
            v = rng.integers(5, 10)
            assert 5 <= v < 10

    def test_integers_single_value(self):
        rng = SecureRNG()
        for _ in range(20):
            v = rng.integers(7, 8)
            assert v == 7

    def test_integers_invalid_range_raises(self):
        rng = SecureRNG()
        with pytest.raises(ValueError, match="high must be > low"):
            rng.integers(5, 5)
        with pytest.raises(ValueError, match="high must be > low"):
            rng.integers(10, 5)

    def test_random_array_shape(self):
        rng = SecureRNG()
        arr = rng.random_array(50)
        assert arr.shape == (50,)
        assert np.all(arr >= 0.0)
        assert np.all(arr < 1.0)

    def test_seeded_deterministic(self):
        """Seeded SecureRNG should be deterministic."""
        seed = b"test_seed_123"
        rng1 = SecureRNG(seed=seed)
        rng2 = SecureRNG(seed=seed)
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 == vals2

    def test_unseeded_varies(self):
        """Two unseeded SecureRNGs should (almost surely) produce different values."""
        rng1 = SecureRNG()
        rng2 = SecureRNG()
        v1 = rng1.random()
        v2 = rng2.random()
        # Probability of collision is astronomically low
        assert v1 != v2

    def test_repr(self):
        assert "os.urandom" in repr(SecureRNG())
        assert "seeded" in repr(SecureRNG(seed=b"abc"))

    def test_uniformity_of_random(self):
        """SecureRNG.random() output should look uniform."""
        rng = SecureRNG(seed=b"uniformity_test")
        samples = np.array([rng.random() for _ in range(5_000)])
        result = uniformity_test(samples, n_bins=20)
        assert result["passed"], f"Uniformity test failed: p={result['p_value']:.4f}"


# ---------------------------------------------------------------------------
# constant_time_sample and sample_with_seed
# ---------------------------------------------------------------------------

class TestConstantTimeSample:
    """Tests for constant_time_sample."""

    def test_valid_outcome(self, skewed_probs):
        rng = SecureRNG(seed=b"ct_test")
        for _ in range(50):
            s = constant_time_sample(skewed_probs, rng=rng)
            assert 0 <= s < len(skewed_probs)

    def test_degenerate_distribution(self):
        p = np.array([0.0, 0.0, 1.0])
        rng = SecureRNG(seed=b"degen")
        for _ in range(20):
            assert constant_time_sample(p, rng=rng) == 2

    def test_uses_default_rng_when_none(self, uniform_probs):
        s = constant_time_sample(uniform_probs)
        assert 0 <= s < len(uniform_probs)


class TestSampleWithSeed:
    """Tests for sample_with_seed."""

    def test_deterministic(self, skewed_probs):
        s1 = sample_with_seed(skewed_probs, seed=42)
        s2 = sample_with_seed(skewed_probs, seed=42)
        assert s1 == s2

    def test_different_seeds(self, uniform_probs):
        """Different seeds should produce different samples (usually)."""
        results = {sample_with_seed(uniform_probs, seed=i) for i in range(100)}
        # With 4 outcomes and 100 tries, we should see multiple outcomes
        assert len(results) > 1

    def test_valid_range(self, skewed_probs):
        for seed in range(50):
            s = sample_with_seed(skewed_probs, seed=seed)
            assert 0 <= s < len(skewed_probs)


# ---------------------------------------------------------------------------
# DiscreteDistribution
# ---------------------------------------------------------------------------

class TestDiscreteDistribution:
    """Tests for DiscreteDistribution."""

    def test_basic_construction(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        assert dist.k == 4
        np.testing.assert_allclose(dist.probs, skewed_probs)

    def test_custom_values(self):
        p = np.array([0.3, 0.7])
        vals = np.array([-1.0, 1.0])
        dist = DiscreteDistribution(p, values=vals)
        np.testing.assert_allclose(dist.values, [-1.0, 1.0])

    def test_mismatched_values_raises(self):
        with pytest.raises(ValueError, match="must match probs length"):
            DiscreteDistribution(
                np.array([0.5, 0.5]),
                values=np.array([1.0, 2.0, 3.0]),
            )

    def test_pmf_in_support(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        for i, p in enumerate(skewed_probs):
            assert dist.pmf(float(i)) == pytest.approx(p)

    def test_pmf_not_in_support(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        assert dist.pmf(99.0) == 0.0

    def test_log_pmf(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        assert dist.log_pmf(0.0) == pytest.approx(math.log(0.1))
        assert dist.log_pmf(99.0) == -np.inf

    def test_cdf(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        assert dist.cdf(-1.0) == pytest.approx(0.0)
        assert dist.cdf(0.0) == pytest.approx(0.1)
        assert dist.cdf(1.0) == pytest.approx(0.3)
        assert dist.cdf(3.0) == pytest.approx(1.0)

    def test_mean(self):
        dist = DiscreteDistribution(
            np.array([0.2, 0.3, 0.5]),
            values=np.array([-1.0, 0.0, 1.0]),
        )
        expected = 0.2 * (-1.0) + 0.3 * 0.0 + 0.5 * 1.0
        assert dist.mean() == pytest.approx(expected)

    def test_variance(self):
        dist = DiscreteDistribution(
            np.array([0.2, 0.3, 0.5]),
            values=np.array([-1.0, 0.0, 1.0]),
        )
        mu = dist.mean()
        ex2 = 0.2 * 1.0 + 0.3 * 0.0 + 0.5 * 1.0
        expected_var = ex2 - mu**2
        assert dist.variance() == pytest.approx(expected_var)

    def test_std(self):
        dist = DiscreteDistribution(np.array([0.5, 0.5]), values=np.array([0.0, 1.0]))
        assert dist.std() == pytest.approx(0.5)

    def test_entropy_uniform(self, uniform_probs):
        dist = DiscreteDistribution(uniform_probs)
        expected = math.log(4.0)
        assert dist.entropy() == pytest.approx(expected)

    def test_entropy_degenerate(self):
        dist = DiscreteDistribution(np.array([1.0, 0.0, 0.0]))
        assert dist.entropy() == pytest.approx(0.0)

    def test_mode(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        assert dist.mode() == pytest.approx(3.0)

    def test_mode_custom_values(self):
        dist = DiscreteDistribution(
            np.array([0.1, 0.9]),
            values=np.array([100.0, 200.0]),
        )
        assert dist.mode() == pytest.approx(200.0)

    def test_support(self):
        dist = DiscreteDistribution(np.array([0.5, 0.0, 0.5]))
        sup = dist.support()
        np.testing.assert_allclose(sorted(sup), [0.0, 2.0])

    def test_sample_shape(self, uniform_probs):
        dist = DiscreteDistribution(uniform_probs)
        rng = np.random.default_rng(42)
        samples = dist.sample(1000, rng=rng)
        assert samples.shape == (1000,)

    def test_sample_values_in_support(self):
        vals = np.array([10.0, 20.0, 30.0])
        dist = DiscreteDistribution(np.array([0.2, 0.3, 0.5]), values=vals)
        rng = np.random.default_rng(42)
        samples = dist.sample(100, rng=rng)
        for s in samples:
            assert s in vals

    def test_sample_distribution(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        rng = np.random.default_rng(42)
        n = 50_000
        samples = dist.sample(n, rng=rng)
        counts = np.array([np.sum(np.isclose(samples, i)) for i in range(4)]) / n
        np.testing.assert_allclose(counts, skewed_probs, atol=0.01)

    def test_repr(self, skewed_probs):
        dist = DiscreteDistribution(skewed_probs)
        r = repr(dist)
        assert "k=4" in r
        assert "mean=" in r


# ---------------------------------------------------------------------------
# MixtureDistribution
# ---------------------------------------------------------------------------

class TestMixtureDistribution:
    """Tests for MixtureDistribution."""

    def test_basic_construction(self):
        d1 = DiscreteDistribution(np.array([0.9, 0.1]))
        d2 = DiscreteDistribution(np.array([0.1, 0.9]))
        mix = MixtureDistribution([d1, d2], weights=np.array([0.5, 0.5]))
        assert len(mix.components) == 2

    def test_empty_components_raises(self):
        with pytest.raises(ValueError, match="At least one component"):
            MixtureDistribution([], weights=np.array([]))

    def test_mismatched_weights_raises(self):
        d1 = DiscreteDistribution(np.array([1.0]))
        with pytest.raises(ValueError, match="must match"):
            MixtureDistribution([d1], weights=np.array([0.5, 0.5]))

    def test_mean(self):
        d1 = DiscreteDistribution(np.array([1.0, 0.0]), values=np.array([0.0, 1.0]))
        d2 = DiscreteDistribution(np.array([0.0, 1.0]), values=np.array([0.0, 1.0]))
        mix = MixtureDistribution([d1, d2], weights=np.array([0.5, 0.5]))
        assert mix.mean() == pytest.approx(0.5)

    def test_variance(self):
        d1 = DiscreteDistribution(np.array([1.0, 0.0]), values=np.array([0.0, 1.0]))
        d2 = DiscreteDistribution(np.array([0.0, 1.0]), values=np.array([0.0, 1.0]))
        mix = MixtureDistribution([d1, d2], weights=np.array([0.5, 0.5]))
        # E[X] = 0.5, E[X^2] = 0.5*0 + 0.5*1 = 0.5, Var = 0.5 - 0.25 = 0.25
        assert mix.variance() == pytest.approx(0.25)

    def test_pmf(self):
        d1 = DiscreteDistribution(np.array([0.8, 0.2]))
        d2 = DiscreteDistribution(np.array([0.2, 0.8]))
        mix = MixtureDistribution([d1, d2], weights=np.array([0.5, 0.5]))
        assert mix.pmf(0.0) == pytest.approx(0.5 * 0.8 + 0.5 * 0.2)
        assert mix.pmf(1.0) == pytest.approx(0.5 * 0.2 + 0.5 * 0.8)

    def test_sample_shape(self):
        d1 = DiscreteDistribution(np.array([0.5, 0.5]))
        mix = MixtureDistribution([d1], weights=np.array([1.0]))
        rng = np.random.default_rng(42)
        samples = mix.sample(500, rng=rng)
        assert samples.shape == (500,)

    def test_sample_distribution(self):
        """Mixture of two Bernoullis should produce correct frequencies."""
        d1 = DiscreteDistribution(np.array([1.0, 0.0]))  # always 0
        d2 = DiscreteDistribution(np.array([0.0, 1.0]))  # always 1
        mix = MixtureDistribution([d1, d2], weights=np.array([0.3, 0.7]))
        rng = np.random.default_rng(42)
        samples = mix.sample(30_000, rng=rng)
        freq_1 = np.mean(np.abs(samples - 1.0) < 0.5)
        assert freq_1 == pytest.approx(0.7, abs=0.02)

    def test_repr(self):
        d1 = DiscreteDistribution(np.array([0.5, 0.5]))
        mix = MixtureDistribution([d1], weights=np.array([1.0]))
        assert "n_components=1" in repr(mix)


# ---------------------------------------------------------------------------
# TruncatedDistribution
# ---------------------------------------------------------------------------

class TestTruncatedDistribution:
    """Tests for TruncatedDistribution."""

    def test_basic_truncation(self):
        base = DiscreteDistribution(
            np.array([0.2, 0.3, 0.3, 0.2]),
            values=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        trunc = TruncatedDistribution(base, low=1.0, high=2.0)
        assert trunc.k == 2

    def test_empty_truncation_raises(self):
        base = DiscreteDistribution(np.array([0.5, 0.5]), values=np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="No outcomes"):
            TruncatedDistribution(base, low=5.0, high=10.0)

    def test_pmf_renormalized(self):
        base = DiscreteDistribution(
            np.array([0.2, 0.3, 0.5]),
            values=np.array([0.0, 1.0, 2.0]),
        )
        trunc = TruncatedDistribution(base, low=1.0, high=2.0)
        # Original mass in [1, 2] = 0.3 + 0.5 = 0.8
        assert trunc.pmf(1.0) == pytest.approx(0.3 / 0.8)
        assert trunc.pmf(2.0) == pytest.approx(0.5 / 0.8)
        assert trunc.pmf(0.0) == 0.0

    def test_mean(self):
        base = DiscreteDistribution(
            np.array([0.25, 0.25, 0.25, 0.25]),
            values=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        trunc = TruncatedDistribution(base, low=2.0, high=3.0)
        assert trunc.mean() == pytest.approx(2.5)

    def test_sample_within_bounds(self):
        base = DiscreteDistribution(
            np.array([0.2, 0.3, 0.3, 0.2]),
            values=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        trunc = TruncatedDistribution(base, low=1.0, high=2.0)
        rng = np.random.default_rng(42)
        samples = trunc.sample(200, rng=rng)
        assert np.all(samples >= 1.0 - 1e-9)
        assert np.all(samples <= 2.0 + 1e-9)

    def test_repr(self):
        base = DiscreteDistribution(np.array([0.5, 0.5]), values=np.array([0.0, 1.0]))
        trunc = TruncatedDistribution(base, low=0.0, high=1.0)
        r = repr(trunc)
        assert "TruncatedDistribution" in r


# ---------------------------------------------------------------------------
# ConditionalDistribution
# ---------------------------------------------------------------------------

class TestConditionalDistribution:
    """Tests for ConditionalDistribution."""

    def test_basic_conditioning(self):
        base = DiscreteDistribution(
            np.array([0.3, 0.3, 0.4]),
            values=np.array([0.0, 1.0, 2.0]),
        )
        cond = ConditionalDistribution(base, condition=lambda x: x > 0.5)
        # Keeps values 1.0 and 2.0, with original mass 0.3 + 0.4 = 0.7
        assert cond.pmf(1.0) == pytest.approx(0.3 / 0.7)
        assert cond.pmf(2.0) == pytest.approx(0.4 / 0.7)
        assert cond.pmf(0.0) == 0.0

    def test_impossible_condition_raises(self):
        base = DiscreteDistribution(np.array([0.5, 0.5]), values=np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="zero probability mass"):
            ConditionalDistribution(base, condition=lambda x: x > 10)

    def test_mean(self):
        base = DiscreteDistribution(
            np.array([0.25, 0.25, 0.25, 0.25]),
            values=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        cond = ConditionalDistribution(base, condition=lambda x: x >= 2.0)
        assert cond.mean() == pytest.approx(2.5)

    def test_conditioning_probability(self):
        base = DiscreteDistribution(
            np.array([0.2, 0.3, 0.5]),
            values=np.array([0.0, 1.0, 2.0]),
        )
        cond = ConditionalDistribution(base, condition=lambda x: x >= 1.0)
        assert cond.conditioning_probability == pytest.approx(0.8)

    def test_sample_satisfies_condition(self):
        base = DiscreteDistribution(
            np.array([0.3, 0.3, 0.4]),
            values=np.array([-1.0, 0.0, 1.0]),
        )
        cond = ConditionalDistribution(base, condition=lambda x: x >= 0)
        rng = np.random.default_rng(42)
        samples = cond.sample(200, rng=rng)
        assert np.all(samples >= -1e-9)

    def test_repr(self):
        base = DiscreteDistribution(np.array([0.5, 0.5]))
        cond = ConditionalDistribution(base, condition=lambda x: x == 0)
        r = repr(cond)
        assert "ConditionalDistribution" in r


# ---------------------------------------------------------------------------
# MechanismSampler
# ---------------------------------------------------------------------------

class TestMechanismSampler:
    """Tests for MechanismSampler."""

    def test_construction(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        assert sampler._effective_method == "alias"

    def test_auto_selects_alias_for_large_k(self):
        p = np.random.default_rng(0).dirichlet(np.ones(50), size=3)
        mech = ExtractedMechanism(p_final=p)
        sampler = MechanismSampler(mech, method="auto")
        assert sampler._effective_method == "alias"

    def test_auto_selects_cdf_for_small_k(self):
        p = np.random.default_rng(0).dirichlet(np.ones(10), size=3)
        mech = ExtractedMechanism(p_final=p)
        sampler = MechanismSampler(mech, method="auto")
        assert sampler._effective_method == "cdf"

    def test_invalid_method_raises(self, simple_mechanism):
        with pytest.raises(ValueError, match="method must be one of"):
            MechanismSampler(simple_mechanism, method="invalid")

    def test_sample_shape(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        result = sampler.sample(true_value=0, n=10)
        assert result.shape == (10,)
        assert result.dtype == np.int64

    def test_sample_single(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        result = sampler.sample(true_value=1, n=1)
        assert result.shape == (1,)
        assert 0 <= result[0] < 5

    def test_sample_valid_range(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="cdf", seed=42)
        samples = sampler.sample(true_value=0, n=100)
        assert np.all(samples >= 0)
        assert np.all(samples < 5)

    def test_sample_values(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        vals = sampler.sample_values(true_value=0, n=10)
        assert vals.shape == (10,)

    def test_sample_vectorized(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        results = sampler.sample_vectorized([0, 1, 2])
        assert results.shape == (3,)

    def test_with_spec(self, simple_mechanism, simple_spec):
        sampler = MechanismSampler(
            simple_mechanism, spec=simple_spec, method="alias", seed=42
        )
        result = sampler.sample(true_value=1.0, n=5)
        assert result.shape == (5,)

    def test_resolve_row_with_spec(self, simple_mechanism, simple_spec):
        sampler = MechanismSampler(
            simple_mechanism, spec=simple_spec, method="alias", seed=42
        )
        # true_value=1.5 should map to closest query value (2.0 at index 2)
        row = sampler._resolve_row(1.5)
        assert row == 1 or row == 2  # either could be closest

    def test_resolve_row_out_of_range(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias")
        with pytest.raises(ValueError, match="out of range"):
            sampler._resolve_row(99)

    def test_estimate_mse(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        mse = sampler.estimate_mse([0, 1], n_samples=1000)
        assert mse >= 0.0

    def test_estimate_mae(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        mae = sampler.estimate_mae([0, 1], n_samples=1000)
        assert mae >= 0.0

    def test_row_distribution(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias")
        dist = sampler.row_distribution(0)
        assert isinstance(dist, DiscreteDistribution)
        assert dist.k == 5
        np.testing.assert_allclose(dist.probs, simple_mechanism.p_final[0])

    def test_row_distribution_out_of_range(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias")
        with pytest.raises(ValueError, match="out of range"):
            sampler.row_distribution(10)

    def test_repr(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias")
        r = repr(sampler)
        assert "n=3" in r
        assert "k=5" in r

    def test_different_methods_sample_correctly(self, simple_mechanism):
        """Both alias and CDF should produce samples from the correct distribution."""
        for method in ("alias", "cdf"):
            sampler = MechanismSampler(simple_mechanism, method=method, seed=42)
            samples = sampler.sample(true_value=0, n=20_000)
            counts = np.bincount(samples, minlength=5) / 20_000
            expected = simple_mechanism.p_final[0]
            np.testing.assert_allclose(counts, expected, atol=0.02)

    def test_caching_row_samplers(self, simple_mechanism):
        """Row samplers should be cached for reuse."""
        sampler = MechanismSampler(simple_mechanism, method="alias", seed=42)
        sampler.sample(true_value=0, n=1)
        assert 0 in sampler._row_samplers
        s1 = sampler._get_row_sampler(0)
        s2 = sampler._get_row_sampler(0)
        assert s1 is s2


# ---------------------------------------------------------------------------
# Privacy audit
# ---------------------------------------------------------------------------

class TestPrivacyAudit:
    """Tests for MechanismSampler.privacy_audit."""

    def test_audit_passes_for_dp_mechanism(self, simple_spec):
        """A mechanism satisfying DP constraints should pass the audit."""
        n, k = 3, 5
        eps = simple_spec.epsilon
        # Build a mechanism that satisfies DP by construction
        rng = np.random.default_rng(42)
        base_row = rng.dirichlet(np.ones(k))
        p = np.tile(base_row, (n, 1))
        # The identical-row mechanism trivially satisfies DP (ratio = 1)
        mech = ExtractedMechanism(p_final=p)
        sampler = MechanismSampler(mech, spec=simple_spec, method="alias", seed=0)
        result = sampler.privacy_audit(n_samples=10_000)
        assert result["passed"]
        assert result["max_log_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_audit_requires_spec(self, simple_mechanism):
        sampler = MechanismSampler(simple_mechanism, method="alias")
        with pytest.raises(RuntimeError, match="requires a QuerySpec"):
            sampler.privacy_audit()

    def test_audit_returns_required_keys(self, simple_mechanism, simple_spec):
        sampler = MechanismSampler(
            simple_mechanism, spec=simple_spec, method="alias", seed=0
        )
        result = sampler.privacy_audit(n_samples=1_000)
        required_keys = {
            "max_ratio", "max_log_ratio", "epsilon_target",
            "passed", "worst_pair", "worst_outcome",
        }
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestChiSquaredTest:
    """Tests for chi_squared_test."""

    def test_passes_for_matching_distribution(self):
        probs = np.array([0.2, 0.3, 0.5])
        rng = np.random.default_rng(42)
        samples = rng.choice(3, size=10_000, p=probs)
        result = chi_squared_test(samples, probs)
        assert result["passed"]
        assert result["p_value"] > 0.01

    def test_fails_for_mismatched_distribution(self):
        probs = np.array([0.5, 0.5])
        # All samples are 0 -> clearly not matching [0.5, 0.5]
        samples = np.zeros(10_000, dtype=np.int64)
        result = chi_squared_test(samples, probs)
        assert not result["passed"]

    def test_result_keys(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        samples = np.array([0, 1, 2, 3] * 100, dtype=np.int64)
        result = chi_squared_test(samples, probs)
        assert "statistic" in result
        assert "df" in result
        assert "p_value" in result
        assert "passed" in result
        assert "n_bins_original" in result
        assert "n_bins_merged" in result
        assert "n_samples" in result

    def test_bin_merging(self):
        """Bins with low expected count should be merged."""
        probs = np.array([0.001, 0.001, 0.998])
        rng = np.random.default_rng(42)
        samples = rng.choice(3, size=500, p=probs)
        result = chi_squared_test(samples, probs)
        assert result["n_bins_merged"] < result["n_bins_original"]


class TestKSTest:
    """Tests for ks_test."""

    def test_passes_for_correct_cdf(self):
        """Samples from uniform [0,1) should pass KS test against uniform CDF."""
        rng = np.random.default_rng(42)
        samples = rng.random(5_000)
        result = ks_test(samples, cdf_fn=lambda x: np.clip(x, 0, 1))
        assert result["passed"]

    def test_fails_for_wrong_cdf(self):
        """Samples from one distribution tested against wrong CDF should fail."""
        rng = np.random.default_rng(42)
        samples = rng.random(5_000)  # Uniform [0, 1)
        # Test against a CDF that's completely off
        result = ks_test(samples, cdf_fn=lambda x: 0.0 if x < 0.9 else 1.0)
        assert not result["passed"]

    def test_result_keys(self):
        samples = np.array([0.1, 0.5, 0.9])
        result = ks_test(samples, cdf_fn=lambda x: x)
        assert "statistic" in result
        assert "n" in result
        assert "critical_value_05" in result
        assert "passed" in result
        assert "d_plus" in result
        assert "d_minus" in result

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ks_test(np.array([]), cdf_fn=lambda x: x)


class TestUniformityTest:
    """Tests for uniformity_test."""

    def test_passes_for_uniform(self):
        rng = np.random.default_rng(42)
        samples = rng.random(10_000)
        result = uniformity_test(samples, n_bins=50)
        assert result["passed"]

    def test_fails_for_non_uniform(self):
        samples = np.full(10_000, 0.5)
        result = uniformity_test(samples, n_bins=50)
        assert not result["passed"]

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            uniformity_test(np.array([]))

    def test_result_keys(self):
        rng = np.random.default_rng(0)
        result = uniformity_test(rng.random(1000), n_bins=20)
        assert "statistic" in result
        assert "p_value" in result
        assert "n_bins" in result
        assert "min_bin_count" in result
        assert "max_bin_count" in result


# ---------------------------------------------------------------------------
# build_sampler factory
# ---------------------------------------------------------------------------

class TestBuildSampler:
    """Tests for build_sampler factory."""

    def test_default_config(self, simple_mechanism):
        sampler = build_sampler(simple_mechanism)
        assert isinstance(sampler, MechanismSampler)

    def test_alias_method(self, simple_mechanism):
        config = SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
        sampler = build_sampler(simple_mechanism, config=config)
        assert sampler._effective_method == "alias"

    def test_cdf_method(self, simple_mechanism):
        config = SamplingConfig(method=SamplingMethod.CDF, seed=42)
        sampler = build_sampler(simple_mechanism, config=config)
        assert sampler._effective_method == "cdf"

    def test_with_spec(self, simple_mechanism, simple_spec):
        config = SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
        sampler = build_sampler(simple_mechanism, spec=simple_spec, config=config)
        assert sampler.spec is simple_spec

    def test_seed_propagation(self, simple_mechanism):
        config = SamplingConfig(method=SamplingMethod.ALIAS, seed=123)
        sampler = build_sampler(simple_mechanism, config=config)
        assert sampler._seed == 123


# ---------------------------------------------------------------------------
# Cross-sampler consistency
# ---------------------------------------------------------------------------

class TestCrossSamplerConsistency:
    """Verify that alias and CDF samplers produce consistent distributions."""

    def test_alias_vs_cdf_chi_squared(self, skewed_probs):
        """Both samplers should pass chi-squared test for the same distribution."""
        alias = AliasMethodSampler()
        alias.build(skewed_probs)
        cdf = InverseCDFSampler()
        cdf.build(skewed_probs)

        n = 50_000
        alias_samples = alias.sample_batch(n, rng=np.random.default_rng(42))
        cdf_samples = cdf.sample_batch(n, rng=np.random.default_rng(42))

        alias_result = chi_squared_test(alias_samples, skewed_probs)
        cdf_result = chi_squared_test(cdf_samples, skewed_probs)

        assert alias_result["passed"]
        assert cdf_result["passed"]

    def test_same_log_probabilities(self, skewed_probs):
        """Both samplers should report identical log-probabilities."""
        alias = AliasMethodSampler()
        alias.build(skewed_probs)
        cdf = InverseCDFSampler()
        cdf.build(skewed_probs)

        for i in range(len(skewed_probs)):
            assert alias.log_probability(i) == pytest.approx(
                cdf.log_probability(i), abs=1e-12
            )


# ---------------------------------------------------------------------------
# Edge cases and stress tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_two_outcome_distribution(self):
        """Minimal non-trivial distribution (k=2)."""
        p = np.array([0.3, 0.7])
        for Cls in (AliasMethodSampler, InverseCDFSampler):
            sampler = Cls()
            sampler.build(p)
            rng = np.random.default_rng(42)
            samples = sampler.sample_batch(10_000, rng=rng)
            freq = np.bincount(samples, minlength=2) / 10_000
            np.testing.assert_allclose(freq, p, atol=0.02)

    def test_near_zero_probabilities(self):
        """Distribution with very small probabilities."""
        k = 100
        p = np.full(k, 1e-4)
        p[0] = 1.0 - (k - 1) * 1e-4
        p = p / p.sum()  # ensure exact normalization
        sampler = AliasMethodSampler()
        sampler.build(p)
        rng = np.random.default_rng(42)
        samples = sampler.sample_batch(50_000, rng=rng)
        assert samples.min() >= 0
        assert samples.max() < k

    def test_highly_skewed_distribution(self):
        """Distribution with almost all mass on one outcome."""
        p = np.array([0.999, 0.0005, 0.0005])
        sampler = AliasMethodSampler()
        sampler.build(p)
        rng = np.random.default_rng(42)
        samples = sampler.sample_batch(10_000, rng=rng)
        assert np.sum(samples == 0) > 9_800

    def test_discrete_distribution_with_negative_values(self):
        """DiscreteDistribution with negative outcome values."""
        dist = DiscreteDistribution(
            np.array([0.25, 0.25, 0.25, 0.25]),
            values=np.array([-2.0, -1.0, 0.0, 1.0]),
        )
        assert dist.mean() == pytest.approx(-0.5)
        assert dist.cdf(-1.5) == pytest.approx(0.25)
        assert dist.pmf(-2.0) == pytest.approx(0.25)

    def test_mechanism_sampler_cdf_method(self):
        """MechanismSampler with explicit CDF method."""
        p = np.random.default_rng(0).dirichlet(np.ones(5), size=2)
        mech = ExtractedMechanism(p_final=p)
        sampler = MechanismSampler(mech, method="cdf", seed=42)
        result = sampler.sample(true_value=0, n=100)
        assert result.shape == (100,)
        assert np.all((result >= 0) & (result < 5))

    def test_mixture_three_components(self):
        """Mixture with three components."""
        d1 = DiscreteDistribution(np.array([1.0, 0.0, 0.0]))
        d2 = DiscreteDistribution(np.array([0.0, 1.0, 0.0]))
        d3 = DiscreteDistribution(np.array([0.0, 0.0, 1.0]))
        mix = MixtureDistribution(
            [d1, d2, d3], weights=np.array([0.2, 0.3, 0.5])
        )
        assert mix.mean() == pytest.approx(0.2 * 0 + 0.3 * 1 + 0.5 * 2)
        rng = np.random.default_rng(42)
        samples = mix.sample(10_000, rng=rng)
        freq = np.bincount(samples.astype(int), minlength=3) / 10_000
        np.testing.assert_allclose(freq, [0.2, 0.3, 0.5], atol=0.02)

    def test_truncated_identity(self):
        """Truncation covering entire support should equal the base distribution."""
        base = DiscreteDistribution(
            np.array([0.3, 0.7]),
            values=np.array([0.0, 1.0]),
        )
        trunc = TruncatedDistribution(base, low=-10.0, high=10.0)
        assert trunc.mean() == pytest.approx(base.mean())
        assert trunc.variance() == pytest.approx(base.variance())

    def test_conditional_everything_accepted(self):
        """Conditioning on always-true predicate should equal base distribution."""
        base = DiscreteDistribution(np.array([0.4, 0.6]))
        cond = ConditionalDistribution(base, condition=lambda x: True)
        assert cond.mean() == pytest.approx(base.mean())
        assert cond.conditioning_probability == pytest.approx(1.0)
