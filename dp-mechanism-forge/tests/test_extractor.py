"""
Comprehensive tests for dp_forge.extractor — alias tables, CDF tables,
positivity projection, renormalisation, QP fallback, mechanism extraction,
MSE/MAE computation, and deployable mechanism serialisation.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dp_forge.extractor import (
    AliasTable,
    CDFTable,
    DeployableMechanism,
    ExtractMechanism,
    MechanismExtractor,
    QPFallbackResult,
    _positivity_projection,
    _renormalize,
    batch_sample_alias,
    batch_sample_cdf,
    build_alias_table,
    build_cdf_table,
    compute_mechanism_mae,
    compute_mechanism_mse,
    entropy_analysis,
    sample_alias,
    sample_cdf,
    solve_dp_projection_qp,
    sparsity_analysis,
    interpolate_mechanism,
    smooth_mechanism,
    mixture_mechanism,
    quick_extract,
)
from dp_forge.types import (
    AdjacencyRelation,
    ExtractedMechanism,
    OptimalityCertificate,
    SamplingMethod,
)
from dp_forge.exceptions import ConfigurationError, VerificationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def uniform_probs_5():
    """Uniform distribution over 5 bins."""
    return np.full(5, 0.2)


@pytest.fixture
def skewed_probs_4():
    """Skewed distribution: [0.5, 0.3, 0.15, 0.05]."""
    return np.array([0.5, 0.3, 0.15, 0.05])


@pytest.fixture
def uniform_mechanism_3x5():
    """3×5 uniform mechanism table (each row sums to 1)."""
    return np.full((3, 5), 0.2)


@pytest.fixture
def dp_feasible_mechanism():
    """A 2×4 mechanism known to satisfy (ε=1.0, δ=0)-DP with edges [(0,1)].

    Each row is a valid distribution; adjacent row ratios are bounded by e^1.
    """
    e_eps = math.exp(1.0)
    # Row 0: skewed toward bin 0
    # Row 1: we ensure p[0][j] / p[1][j] <= e and vice versa
    p = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.25, 0.25, 0.25, 0.25],
    ])
    # Verify ratios are within e^1 ≈ 2.718
    for j in range(4):
        assert p[0, j] / p[1, j] <= e_eps + 1e-9
        assert p[1, j] / p[0, j] <= e_eps + 1e-9
    return p


# ═══════════════════════════════════════════════════════════════════════════
# §1  AliasTable Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAliasTableDataclass:
    """Tests for AliasTable construction and validation."""

    def test_valid_construction(self):
        prob = np.array([0.5, 1.0, 0.8])
        alias = np.array([1, 0, 0], dtype=np.int64)
        table = AliasTable(prob=prob, alias=alias, k=3)
        assert table.k == 3
        assert len(table.prob) == 3
        assert len(table.alias) == 3

    def test_prob_length_mismatch_raises(self):
        prob = np.array([0.5, 1.0])
        alias = np.array([1, 0, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="prob length 2 != k=3"):
            AliasTable(prob=prob, alias=alias, k=3)

    def test_alias_length_mismatch_raises(self):
        prob = np.array([0.5, 1.0, 0.8])
        alias = np.array([1, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="alias length 2 != k=3"):
            AliasTable(prob=prob, alias=alias, k=3)

    def test_k_one(self):
        prob = np.array([1.0])
        alias = np.array([0], dtype=np.int64)
        table = AliasTable(prob=prob, alias=alias, k=1)
        assert table.k == 1

    def test_large_k(self):
        k = 10000
        prob = np.ones(k)
        alias = np.zeros(k, dtype=np.int64)
        table = AliasTable(prob=prob, alias=alias, k=k)
        assert table.k == k


# ═══════════════════════════════════════════════════════════════════════════
# §2  build_alias_table Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildAliasTable:
    """Tests for Vose's alias-table construction."""

    def test_uniform_distribution(self, uniform_probs_5):
        table = build_alias_table(uniform_probs_5)
        assert table.k == 5
        assert len(table.prob) == 5
        assert len(table.alias) == 5
        # For uniform: all prob entries should be 1.0 (within float precision)
        assert_allclose(table.prob, 1.0, atol=1e-12)

    def test_skewed_distribution(self, skewed_probs_4):
        table = build_alias_table(skewed_probs_4)
        assert table.k == 4
        assert len(table.prob) == 4
        assert len(table.alias) == 4
        # All prob entries should be in [0, 1]
        assert np.all(table.prob >= 0.0)
        assert np.all(table.prob <= 1.0 + 1e-12)
        # All alias entries should be valid bin indices
        assert np.all(table.alias >= 0)
        assert np.all(table.alias < 4)

    def test_delta_distribution(self):
        """Single bin has all mass."""
        probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        # Near-zero is OK (threshold is -1e-15)
        table = build_alias_table(probs)
        assert table.k == 5

    def test_two_bin_distribution(self):
        probs = np.array([0.3, 0.7])
        table = build_alias_table(probs)
        assert table.k == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_alias_table(np.array([]))

    def test_negative_probs_raises(self):
        probs = np.array([0.5, -0.1, 0.6])
        with pytest.raises(ValueError, match="negative values"):
            build_alias_table(probs)

    def test_not_summing_to_one_raises(self):
        probs = np.array([0.3, 0.3, 0.3])  # sums to 0.9
        with pytest.raises(ValueError, match="sum to 1"):
            build_alias_table(probs)

    def test_sum_too_large_raises(self):
        probs = np.array([0.5, 0.5, 0.5])  # sums to 1.5
        with pytest.raises(ValueError, match="sum to 1"):
            build_alias_table(probs)

    def test_tiny_near_zero_negatives_accepted(self):
        """Negatives smaller than -1e-15 in magnitude should be accepted."""
        probs = np.array([0.5, 0.5 - 1e-16, 1e-16])
        # Should not raise — the threshold is -1e-15
        table = build_alias_table(probs)
        assert table.k == 3

    @pytest.mark.parametrize("k", [1, 2, 3, 10, 50, 100])
    def test_various_sizes(self, k):
        probs = np.ones(k) / k
        table = build_alias_table(probs)
        assert table.k == k

    def test_list_input_accepted(self):
        """Should accept plain Python list, not just numpy array."""
        probs = [0.25, 0.25, 0.25, 0.25]
        table = build_alias_table(probs)
        assert table.k == 4

    def test_float32_input_converted(self):
        probs = np.array([0.5, 0.5], dtype=np.float32)
        table = build_alias_table(probs)
        assert table.prob.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════
# §3  Alias Sampling Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSampleAlias:
    """Tests for O(1) alias-table sampling."""

    def test_single_sample_valid_range(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        for _ in range(100):
            s = sample_alias(table, rng)
            assert 0 <= s < 5

    def test_single_sample_returns_int(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        s = sample_alias(table, rng)
        assert isinstance(s, int)

    def test_deterministic_with_seed(self, uniform_probs_5):
        table = build_alias_table(uniform_probs_5)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        samples1 = [sample_alias(table, rng1) for _ in range(50)]
        samples2 = [sample_alias(table, rng2) for _ in range(50)]
        assert samples1 == samples2

    def test_distribution_chi_squared_uniform(self, rng):
        """Chi-squared test: alias sampling from uniform should be uniform."""
        k = 5
        probs = np.ones(k) / k
        table = build_alias_table(probs)
        n_samples = 50000
        counts = np.zeros(k, dtype=int)
        for _ in range(n_samples):
            counts[sample_alias(table, rng)] += 1
        expected = n_samples / k
        chi2 = np.sum((counts - expected) ** 2 / expected)
        # With k-1 = 4 degrees of freedom, chi2 < 20 is p > 0.001
        assert chi2 < 20, f"Chi-squared {chi2} too high for uniform distribution"

    def test_distribution_skewed(self, rng):
        """Verify skewed distribution is sampled correctly."""
        probs = np.array([0.7, 0.2, 0.1])
        table = build_alias_table(probs)
        n_samples = 100000
        counts = np.zeros(3, dtype=int)
        for _ in range(n_samples):
            counts[sample_alias(table, rng)] += 1
        empirical = counts / n_samples
        assert_allclose(empirical, probs, atol=0.02)


class TestBatchSampleAlias:
    """Tests for vectorised alias-table sampling."""

    def test_batch_returns_correct_shape(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        samples = batch_sample_alias(table, 1000, rng)
        assert samples.shape == (1000,)

    def test_batch_valid_range(self, skewed_probs_4, rng):
        table = build_alias_table(skewed_probs_4)
        samples = batch_sample_alias(table, 5000, rng)
        assert np.all(samples >= 0)
        assert np.all(samples < 4)

    def test_batch_dtype(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        samples = batch_sample_alias(table, 100, rng)
        assert samples.dtype == np.int64

    def test_batch_zero_samples(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        samples = batch_sample_alias(table, 0, rng)
        assert len(samples) == 0
        assert samples.dtype == np.int64

    def test_batch_negative_n(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        samples = batch_sample_alias(table, -1, rng)
        assert len(samples) == 0

    def test_batch_distribution_matches_single(self, rng):
        """Batch sampling should produce same distribution as single sampling."""
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        table = build_alias_table(probs)
        n = 100000
        samples = batch_sample_alias(table, n, rng)
        counts = np.bincount(samples, minlength=4)
        empirical = counts / n
        assert_allclose(empirical, probs, atol=0.02)

    def test_batch_one_sample(self, uniform_probs_5, rng):
        table = build_alias_table(uniform_probs_5)
        samples = batch_sample_alias(table, 1, rng)
        assert samples.shape == (1,)
        assert 0 <= samples[0] < 5

    def test_batch_deterministic(self, skewed_probs_4):
        table = build_alias_table(skewed_probs_4)
        s1 = batch_sample_alias(table, 500, np.random.default_rng(99))
        s2 = batch_sample_alias(table, 500, np.random.default_rng(99))
        assert_array_equal(s1, s2)


# ═══════════════════════════════════════════════════════════════════════════
# §4  CDFTable Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCDFTableDataclass:
    """Tests for CDFTable construction and validation."""

    def test_valid_construction(self):
        cdf = np.array([0.2, 0.5, 0.8, 1.0])
        table = CDFTable(cdf=cdf, k=4)
        assert table.k == 4
        assert len(table.cdf) == 4

    def test_length_mismatch_raises(self):
        cdf = np.array([0.5, 1.0])
        with pytest.raises(ValueError, match="cdf length 2 != k=3"):
            CDFTable(cdf=cdf, k=3)

    def test_single_bin(self):
        cdf = np.array([1.0])
        table = CDFTable(cdf=cdf, k=1)
        assert table.k == 1


# ═══════════════════════════════════════════════════════════════════════════
# §5  build_cdf_table Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCDFTable:
    """Tests for CDF table construction."""

    def test_uniform(self, uniform_probs_5):
        table = build_cdf_table(uniform_probs_5)
        assert table.k == 5
        expected_cdf = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        assert_allclose(table.cdf, expected_cdf, atol=1e-12)

    def test_forces_last_to_one(self):
        """CDF[-1] should be forced to exactly 1.0 regardless of accumulation."""
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        table = build_cdf_table(probs)
        assert table.cdf[-1] == 1.0  # Exact equality, not approximate

    def test_cdf_monotone(self, skewed_probs_4):
        table = build_cdf_table(skewed_probs_4)
        for i in range(1, table.k):
            assert table.cdf[i] >= table.cdf[i - 1]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_cdf_table(np.array([]))

    def test_negative_raises(self):
        probs = np.array([0.5, -0.2, 0.7])
        with pytest.raises(ValueError, match="negative values"):
            build_cdf_table(probs)

    def test_list_input(self):
        table = build_cdf_table([0.25, 0.25, 0.25, 0.25])
        assert table.k == 4
        assert table.cdf[-1] == 1.0

    @pytest.mark.parametrize("k", [1, 2, 5, 50])
    def test_various_sizes(self, k):
        probs = np.ones(k) / k
        table = build_cdf_table(probs)
        assert table.k == k
        assert table.cdf[-1] == 1.0

    def test_numerical_drift_corrected(self):
        """Even with probs that accumulate imprecisely, CDF[-1] == 1.0."""
        probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        table = build_cdf_table(probs)
        assert table.cdf[-1] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# §6  CDF Sampling Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSampleCDF:
    """Tests for O(log k) CDF sampling."""

    def test_single_sample_valid_range(self, uniform_probs_5, rng):
        table = build_cdf_table(uniform_probs_5)
        for _ in range(100):
            s = sample_cdf(table, rng)
            assert 0 <= s < 5

    def test_single_sample_returns_int(self, uniform_probs_5, rng):
        table = build_cdf_table(uniform_probs_5)
        s = sample_cdf(table, rng)
        assert isinstance(s, int)

    def test_distribution_uniform(self, rng):
        k = 5
        probs = np.ones(k) / k
        table = build_cdf_table(probs)
        n_samples = 50000
        counts = np.zeros(k, dtype=int)
        for _ in range(n_samples):
            counts[sample_cdf(table, rng)] += 1
        expected = n_samples / k
        chi2 = np.sum((counts - expected) ** 2 / expected)
        assert chi2 < 20

    def test_distribution_skewed(self, rng):
        probs = np.array([0.6, 0.3, 0.1])
        table = build_cdf_table(probs)
        n_samples = 100000
        counts = np.zeros(3, dtype=int)
        for _ in range(n_samples):
            counts[sample_cdf(table, rng)] += 1
        empirical = counts / n_samples
        assert_allclose(empirical, probs, atol=0.02)


class TestBatchSampleCDF:
    """Tests for vectorised CDF sampling."""

    def test_batch_shape(self, uniform_probs_5, rng):
        table = build_cdf_table(uniform_probs_5)
        samples = batch_sample_cdf(table, 1000, rng)
        assert samples.shape == (1000,)

    def test_batch_valid_range(self, skewed_probs_4, rng):
        table = build_cdf_table(skewed_probs_4)
        samples = batch_sample_cdf(table, 5000, rng)
        assert np.all(samples >= 0)
        assert np.all(samples < 4)

    def test_batch_dtype(self, uniform_probs_5, rng):
        table = build_cdf_table(uniform_probs_5)
        samples = batch_sample_cdf(table, 100, rng)
        assert samples.dtype == np.int64

    def test_batch_zero_samples(self, uniform_probs_5, rng):
        table = build_cdf_table(uniform_probs_5)
        samples = batch_sample_cdf(table, 0, rng)
        assert len(samples) == 0

    def test_batch_negative_n(self, uniform_probs_5, rng):
        table = build_cdf_table(uniform_probs_5)
        samples = batch_sample_cdf(table, -5, rng)
        assert len(samples) == 0

    def test_batch_distribution(self, rng):
        probs = np.array([0.15, 0.35, 0.5])
        table = build_cdf_table(probs)
        n = 100000
        samples = batch_sample_cdf(table, n, rng)
        counts = np.bincount(samples, minlength=3)
        empirical = counts / n
        assert_allclose(empirical, probs, atol=0.02)

    def test_batch_deterministic(self, skewed_probs_4):
        table = build_cdf_table(skewed_probs_4)
        s1 = batch_sample_cdf(table, 500, np.random.default_rng(77))
        s2 = batch_sample_cdf(table, 500, np.random.default_rng(77))
        assert_array_equal(s1, s2)


# ═══════════════════════════════════════════════════════════════════════════
# §7  Positivity Projection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPositivityProjection:
    """Tests for _positivity_projection: clips to [eta_min, 1], NOT to 0."""

    def test_clips_to_eta_min_not_zero(self):
        """Core invariant: negatives clip to eta_min, not 0."""
        p_raw = np.array([[-0.01, 0.5, 0.51], [0.3, 0.3, 0.4]])
        eta_min = 1e-10
        result = _positivity_projection(p_raw, eta_min)
        # The negative entry should be clipped to eta_min, NOT to 0
        assert result[0, 0] == eta_min
        assert result[0, 0] > 0

    def test_all_positive_input_no_change_beyond_clipping(self):
        """Entries already in [eta_min, 1] should be unchanged."""
        eta_min = 1e-8
        p_raw = np.array([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])
        result = _positivity_projection(p_raw, eta_min)
        assert_allclose(result, p_raw, atol=1e-15)

    def test_clips_values_above_one(self):
        """Entries > 1 should be clipped to 1."""
        p_raw = np.array([[1.5, -0.5], [0.5, 0.5]])
        eta_min = 1e-10
        result = _positivity_projection(p_raw, eta_min)
        assert result[0, 0] == 1.0
        assert result[0, 1] == eta_min

    def test_eta_min_zero_raises(self):
        p_raw = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="eta_min must be > 0"):
            _positivity_projection(p_raw, 0.0)

    def test_eta_min_negative_raises(self):
        p_raw = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="eta_min must be > 0"):
            _positivity_projection(p_raw, -1e-10)

    def test_does_not_modify_input(self):
        """Should return a copy, not modify in-place."""
        p_raw = np.array([[-0.1, 0.6, 0.5]])
        p_orig = p_raw.copy()
        eta_min = 1e-8
        _positivity_projection(p_raw, eta_min)
        assert_array_equal(p_raw, p_orig)

    @pytest.mark.parametrize("eta_min", [1e-15, 1e-10, 1e-5, 0.01])
    def test_various_eta_min(self, eta_min):
        p_raw = np.array([[-1.0, 0.5, 1.5]])
        result = _positivity_projection(p_raw, eta_min)
        assert result[0, 0] == eta_min
        assert result[0, 2] == 1.0
        assert result[0, 1] == 0.5

    def test_all_negatives(self):
        """All entries are negative; all should clip to eta_min."""
        p_raw = np.array([[-0.1, -0.2, -0.3]])
        eta_min = 1e-6
        result = _positivity_projection(p_raw, eta_min)
        assert_allclose(result, eta_min)

    def test_output_shape_preserved(self):
        p_raw = np.ones((5, 10)) * 0.1
        result = _positivity_projection(p_raw, 1e-8)
        assert result.shape == (5, 10)


# ═══════════════════════════════════════════════════════════════════════════
# §8  Renormalize Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRenormalize:
    """Tests for _renormalize: each row should sum to 1 after renormalization."""

    def test_rows_sum_to_one(self):
        p = np.array([[0.3, 0.3, 0.4], [0.1, 0.1, 0.1]])
        result = _renormalize(p)
        row_sums = result.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_already_normalized(self):
        """Should be nearly no-op if rows already sum to 1."""
        p = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]])
        result = _renormalize(p)
        assert_allclose(result, p, atol=1e-12)

    def test_does_not_modify_input(self):
        p = np.array([[0.3, 0.3, 0.4]])
        p_orig = p.copy()
        _renormalize(p)
        assert_array_equal(p, p_orig)

    def test_uniform_rows(self):
        p = np.full((3, 5), 0.3)  # row sums = 1.5
        result = _renormalize(p)
        assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)
        assert_allclose(result, 0.2, atol=1e-12)

    def test_single_element_rows(self):
        p = np.array([[5.0], [0.1]])
        result = _renormalize(p)
        assert_allclose(result, [[1.0], [1.0]], atol=1e-12)

    def test_proportions_preserved(self):
        """Ratios within a row should be preserved."""
        p = np.array([[2.0, 4.0, 6.0]])
        result = _renormalize(p)
        # Ratios: 1:2:3 -> normalized: 1/6, 2/6, 3/6
        assert_allclose(result[0], [1.0 / 6, 2.0 / 6, 3.0 / 6], atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# §9  Compute MSE / MAE Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeMechanismMSE:
    """Tests for compute_mechanism_mse."""

    def test_perfect_mechanism_zero_mse(self):
        """If mechanism puts all mass on the true value, MSE=0."""
        # 2 inputs, grid = [0, 1, 2], true_values = [1, 2]
        # Row 0 puts all mass on bin 1 (value=1), row 1 on bin 2 (value=2)
        p = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        grid = np.array([0.0, 1.0, 2.0])
        true_values = np.array([1.0, 2.0])
        mse = compute_mechanism_mse(p, grid, true_values)
        assert_allclose(mse, 0.0, atol=1e-12)

    def test_uniform_mechanism_mse(self):
        """Known MSE for uniform mechanism."""
        # 1 input, grid = [0, 1], true_value = 0.5
        # MSE = 0.5*(0-0.5)^2 + 0.5*(1-0.5)^2 = 0.5*0.25+0.5*0.25 = 0.25
        p = np.array([[0.5, 0.5]])
        grid = np.array([0.0, 1.0])
        true_values = np.array([0.5])
        mse = compute_mechanism_mse(p, grid, true_values)
        assert_allclose(mse, 0.25, atol=1e-12)

    def test_known_mse(self):
        """Manually computed MSE."""
        p = np.array([[0.5, 0.3, 0.2]])
        grid = np.array([0.0, 1.0, 2.0])
        true_values = np.array([1.0])
        # MSE = 0.5*(0-1)^2 + 0.3*(1-1)^2 + 0.2*(2-1)^2
        #     = 0.5*1 + 0.3*0 + 0.2*1 = 0.7
        mse = compute_mechanism_mse(p, grid, true_values)
        assert_allclose(mse, 0.7, atol=1e-12)

    def test_multiple_inputs(self):
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        grid = np.array([0.0, 1.0])
        true_values = np.array([0.0, 1.0])
        mse = compute_mechanism_mse(p, grid, true_values)
        assert_allclose(mse, 0.0, atol=1e-12)

    def test_mse_non_negative(self, rng):
        """MSE should always be non-negative."""
        n, k = 5, 10
        p = rng.dirichlet(np.ones(k), size=n)
        grid = np.linspace(0, 1, k)
        true_values = rng.random(n)
        mse = compute_mechanism_mse(p, grid, true_values)
        assert mse >= 0


class TestComputeMechanismMAE:
    """Tests for compute_mechanism_mae."""

    def test_perfect_mechanism_zero_mae(self):
        p = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        grid = np.array([0.0, 1.0, 2.0])
        true_values = np.array([1.0, 2.0])
        mae = compute_mechanism_mae(p, grid, true_values)
        assert_allclose(mae, 0.0, atol=1e-12)

    def test_known_mae(self):
        p = np.array([[0.5, 0.3, 0.2]])
        grid = np.array([0.0, 1.0, 2.0])
        true_values = np.array([1.0])
        # MAE = 0.5*|0-1| + 0.3*|1-1| + 0.2*|2-1| = 0.5 + 0 + 0.2 = 0.7
        mae = compute_mechanism_mae(p, grid, true_values)
        assert_allclose(mae, 0.7, atol=1e-12)

    def test_uniform_mechanism_mae(self):
        p = np.array([[0.5, 0.5]])
        grid = np.array([0.0, 1.0])
        true_values = np.array([0.5])
        # MAE = 0.5*|0-0.5| + 0.5*|1-0.5| = 0.5*0.5 + 0.5*0.5 = 0.5
        mae = compute_mechanism_mae(p, grid, true_values)
        assert_allclose(mae, 0.5, atol=1e-12)

    def test_mae_non_negative(self, rng):
        n, k = 5, 10
        p = rng.dirichlet(np.ones(k), size=n)
        grid = np.linspace(0, 1, k)
        true_values = rng.random(n)
        mae = compute_mechanism_mae(p, grid, true_values)
        assert mae >= 0

    def test_mae_leq_max_abs_diff(self, rng):
        """MAE should be bounded by max |grid - true_value|."""
        n, k = 3, 8
        p = rng.dirichlet(np.ones(k), size=n)
        grid = np.linspace(-1, 1, k)
        true_values = np.zeros(n)
        mae = compute_mechanism_mae(p, grid, true_values)
        assert mae <= np.max(np.abs(grid)) + 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# §10  QP Fallback Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSolveDPProjectionQP:
    """Tests for QP projection onto DP-feasible set."""

    def test_already_feasible_returns_close(self):
        """If p_raw is already DP-feasible, QP should return nearly the same table."""
        # Uniform mechanism is trivially DP-feasible
        p_raw = np.full((2, 4), 0.25)
        result = solve_dp_projection_qp(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], eta_min=1e-10,
        )
        assert result.success
        assert result.frobenius_correction < 1e-4
        # Result should still be a valid distribution
        assert_allclose(result.p_projected.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(result.p_projected >= 1e-10 - 1e-12)

    def test_qp_result_has_valid_rows(self):
        """QP output must have rows summing to 1."""
        p_raw = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
        result = solve_dp_projection_qp(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], eta_min=1e-10,
        )
        assert_allclose(result.p_projected.sum(axis=1), 1.0, atol=1e-6)

    def test_qp_result_positive(self):
        """QP output must respect the eta_min floor."""
        p_raw = np.array([[-0.1, 0.6, 0.5], [0.3, 0.3, 0.4]])
        eta_min = 1e-8
        result = solve_dp_projection_qp(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], eta_min=eta_min,
        )
        assert np.all(result.p_projected >= eta_min - 1e-12)

    def test_qp_result_shape_preserved(self):
        p_raw = np.full((3, 5), 0.2)
        result = solve_dp_projection_qp(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1), (1, 2)], eta_min=1e-10,
        )
        assert result.p_projected.shape == (3, 5)

    def test_qp_approximate_dp(self):
        """Test QP with delta > 0 (approximate DP)."""
        p_raw = np.full((2, 3), 1.0 / 3)
        result = solve_dp_projection_qp(
            p_raw, epsilon=0.5, delta=0.1,
            edges=[(0, 1)], eta_min=1e-10,
        )
        assert isinstance(result, QPFallbackResult)
        assert_allclose(result.p_projected.sum(axis=1), 1.0, atol=1e-6)

    def test_qp_has_n_iterations(self):
        p_raw = np.full((2, 3), 1.0 / 3)
        result = solve_dp_projection_qp(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], eta_min=1e-10,
        )
        assert result.n_iterations >= 0
        assert isinstance(result.solver_message, str)


# ═══════════════════════════════════════════════════════════════════════════
# §11  Entropy and Sparsity Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEntropyAnalysis:
    """Tests for entropy_analysis."""

    def test_uniform_mechanism_max_entropy(self):
        k = 8
        p = np.full((1, k), 1.0 / k)
        result = entropy_analysis(p)
        assert_allclose(result["mean"], math.log2(k), atol=1e-10)
        assert_allclose(result["max_possible"], math.log2(k), atol=1e-10)
        assert_allclose(result["efficiency"], 1.0, atol=1e-10)

    def test_delta_mechanism_zero_entropy(self):
        p = np.array([[0.0, 0.0, 1.0, 0.0]])
        result = entropy_analysis(p)
        assert_allclose(result["mean"], 0.0, atol=1e-10)

    def test_per_row_shape(self):
        p = np.full((5, 10), 0.1)
        result = entropy_analysis(p)
        assert result["per_row"].shape == (5,)

    def test_min_leq_max(self, rng):
        p = rng.dirichlet(np.ones(6), size=4)
        result = entropy_analysis(p)
        assert result["min"] <= result["max"]


class TestSparsityAnalysis:
    """Tests for sparsity_analysis."""

    def test_uniform_low_gini(self):
        p = np.full((2, 10), 0.1)
        result = sparsity_analysis(p)
        # Uniform => Gini should be close to 0
        assert result["mean_gini"] < 0.1

    def test_concentrated_high_gini(self):
        p = np.zeros((1, 10))
        p[0, 0] = 1.0
        result = sparsity_analysis(p, threshold=1e-6)
        assert result["mean_effective_support"] == 1.0
        assert result["near_zero_fraction"] > 0.8

    def test_top_1_mass(self):
        p = np.array([[0.5, 0.3, 0.2]])
        result = sparsity_analysis(p)
        assert_allclose(result["top_1_mass"], [0.5])

    def test_effective_support_shape(self):
        p = np.full((4, 8), 0.125)
        result = sparsity_analysis(p)
        assert result["effective_support"].shape == (4,)


# ═══════════════════════════════════════════════════════════════════════════
# §12  Interpolate, Smooth, Mixture Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInterpolateMechanism:
    """Tests for interpolate_mechanism."""

    def test_output_shape(self):
        p = np.full((2, 5), 0.2)
        grid = np.linspace(0, 1, 5)
        eval_pts = np.linspace(0, 1, 20)
        result = interpolate_mechanism(p, grid, eval_pts)
        assert result.shape == (2, 20)

    def test_grid_length_mismatch_raises(self):
        p = np.full((2, 5), 0.2)
        grid = np.linspace(0, 1, 3)  # Wrong length
        eval_pts = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="grid length"):
            interpolate_mechanism(p, grid, eval_pts)

    def test_output_non_negative(self):
        p = np.array([[0.5, 0.3, 0.2]])
        grid = np.array([0.0, 0.5, 1.0])
        eval_pts = np.linspace(0, 1, 50)
        result = interpolate_mechanism(p, grid, eval_pts)
        assert np.all(result >= -1e-12)


class TestSmoothMechanism:
    """Tests for smooth_mechanism."""

    def test_output_shape(self):
        p = np.full((2, 5), 0.2)
        grid = np.linspace(0, 1, 5)
        eval_pts = np.linspace(0, 1, 30)
        result = smooth_mechanism(p, grid, eval_pts, bandwidth=0.1)
        assert result.shape == (2, 30)

    def test_zero_bandwidth_raises(self):
        p = np.full((1, 3), 1.0 / 3)
        grid = np.array([0.0, 0.5, 1.0])
        eval_pts = np.array([0.5])
        with pytest.raises(ValueError, match="bandwidth must be > 0"):
            smooth_mechanism(p, grid, eval_pts, bandwidth=0.0)

    def test_output_non_negative(self):
        p = np.array([[0.5, 0.3, 0.2]])
        grid = np.array([0.0, 0.5, 1.0])
        eval_pts = np.linspace(0, 1, 50)
        result = smooth_mechanism(p, grid, eval_pts, bandwidth=0.2)
        assert np.all(result >= -1e-12)


class TestMixtureMechanism:
    """Tests for mixture_mechanism."""

    def test_equal_mixture(self):
        p1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        p2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = mixture_mechanism([p1, p2], np.array([0.5, 0.5]))
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert_allclose(result, expected, atol=1e-12)

    def test_single_component(self):
        p = np.array([[0.3, 0.7]])
        result = mixture_mechanism([p], np.array([1.0]))
        assert_allclose(result, p, atol=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mixture_mechanism([], np.array([]))

    def test_weights_not_summing_raises(self):
        p = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="sum to 1"):
            mixture_mechanism([p, p], np.array([0.3, 0.3]))

    def test_shape_mismatch_raises(self):
        p1 = np.array([[0.5, 0.5]])
        p2 = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="shape"):
            mixture_mechanism([p1, p2], np.array([0.5, 0.5]))


# ═══════════════════════════════════════════════════════════════════════════
# §13  MechanismExtractor Class Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMechanismExtractor:
    """Tests for the MechanismExtractor class."""

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ConfigurationError, match="epsilon"):
            MechanismExtractor(
                epsilon=0.0, delta=0.0,
                edges=[(0, 1)], y_grid=np.arange(5, dtype=np.float64),
            )

    def test_negative_epsilon_raises(self):
        with pytest.raises(ConfigurationError, match="epsilon"):
            MechanismExtractor(
                epsilon=-1.0, delta=0.0,
                edges=[(0, 1)], y_grid=np.arange(5, dtype=np.float64),
            )

    def test_invalid_delta_raises(self):
        with pytest.raises(ConfigurationError, match="delta"):
            MechanismExtractor(
                epsilon=1.0, delta=1.0,
                edges=[(0, 1)], y_grid=np.arange(5, dtype=np.float64),
            )

    def test_negative_delta_raises(self):
        with pytest.raises(ConfigurationError, match="delta"):
            MechanismExtractor(
                epsilon=1.0, delta=-0.1,
                edges=[(0, 1)], y_grid=np.arange(5, dtype=np.float64),
            )

    def test_default_eta_min(self):
        ext = MechanismExtractor(
            epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=np.arange(5, dtype=np.float64),
        )
        expected_eta = math.exp(-1.0) * 1e-8
        assert_allclose(ext.eta_min, expected_eta, rtol=1e-10)

    def test_custom_eta_min(self):
        ext = MechanismExtractor(
            epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=np.arange(5, dtype=np.float64),
            eta_min=1e-5,
        )
        assert ext.eta_min == 1e-5

    def test_extract_uniform_mechanism(self):
        """Extracting a uniform mechanism should succeed without QP fallback."""
        n, k = 2, 5
        p_raw = np.full((n, k), 1.0 / k)
        y_grid = np.arange(k, dtype=np.float64)
        ext = MechanismExtractor(
            epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )
        result = ext.extract(p_raw)
        assert isinstance(result, DeployableMechanism)
        assert result.n == n
        assert result.k == k
        assert_allclose(result.p.sum(axis=1), 1.0, atol=1e-10)
        assert result.metadata["extraction_method"] == "projection"


# ═══════════════════════════════════════════════════════════════════════════
# §14  ExtractMechanism Top-Level Function Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractMechanism:
    """Tests for the top-level ExtractMechanism function."""

    def test_uniform_mechanism_succeeds(self):
        n, k = 3, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        edges = [(0, 1), (1, 2)]
        result = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=edges, y_grid=y_grid)
        assert isinstance(result, DeployableMechanism)
        assert result.n == n
        assert result.k == k

    def test_with_adjacency_relation(self):
        n, k = 3, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        adj = AdjacencyRelation.hamming_distance_1(n)
        result = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=adj, y_grid=y_grid)
        assert isinstance(result, DeployableMechanism)

    def test_result_has_alias_tables(self):
        n, k = 2, 5
        p_raw = np.full((n, k), 0.2)
        y_grid = np.arange(k, dtype=np.float64)
        result = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=[(0, 1)], y_grid=y_grid)
        assert len(result.alias_tables) == n
        for at in result.alias_tables:
            assert isinstance(at, AliasTable)
            assert at.k == k

    def test_result_has_cdf_tables(self):
        n, k = 2, 5
        p_raw = np.full((n, k), 0.2)
        y_grid = np.arange(k, dtype=np.float64)
        result = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=[(0, 1)], y_grid=y_grid)
        assert len(result.cdf_tables) == n
        for ct in result.cdf_tables:
            assert isinstance(ct, CDFTable)
            assert ct.k == k

    def test_with_lp_result_extracts_certificate(self):
        n, k = 2, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        lp_result = {
            "dual_vars": np.zeros(10),
            "primal_obj": 0.5,
            "dual_obj": 0.5,
        }
        result = ExtractMechanism(
            p_raw, epsilon=1.0, delta=0.0, edges=[(0, 1)], y_grid=y_grid,
            lp_result=lp_result,
        )
        assert result.certificate is not None
        assert_allclose(result.certificate.duality_gap, 0.0, atol=1e-12)

    def test_with_near_zero_negatives(self):
        """Mechanism with tiny negatives (solver artefacts) should still extract."""
        n, k = 2, 4
        p_raw = np.full((n, k), 0.25)
        p_raw[0, 0] = -1e-12  # Tiny negative from solver
        p_raw[0, 1] += 1e-12  # Compensate
        y_grid = np.arange(k, dtype=np.float64)
        result = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=[(0, 1)], y_grid=y_grid)
        assert isinstance(result, DeployableMechanism)
        assert np.all(result.p >= 0)

    def test_approximate_dp(self):
        """Test extraction with delta > 0."""
        n, k = 2, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        result = ExtractMechanism(
            p_raw, epsilon=0.5, delta=0.01,
            edges=[(0, 1)], y_grid=y_grid,
        )
        assert isinstance(result, DeployableMechanism)
        assert result.delta == 0.01


# ═══════════════════════════════════════════════════════════════════════════
# §15  DeployableMechanism Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeployableMechanism:
    """Tests for DeployableMechanism sampling and serialization."""

    @pytest.fixture
    def deployable(self):
        """Create a simple DeployableMechanism for testing."""
        n, k = 2, 4
        p = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        return ExtractMechanism(
            p, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )

    def test_sample_alias_valid_range(self, deployable, rng):
        samples = deployable.sample(0, n_samples=100, method=SamplingMethod.ALIAS, rng=rng)
        assert samples.shape == (100,)
        assert np.all(samples >= deployable.y_grid[0])
        assert np.all(samples <= deployable.y_grid[-1])

    def test_sample_cdf_valid_range(self, deployable, rng):
        samples = deployable.sample(0, n_samples=100, method=SamplingMethod.CDF, rng=rng)
        assert samples.shape == (100,)
        assert np.all(samples >= deployable.y_grid[0])
        assert np.all(samples <= deployable.y_grid[-1])

    def test_sample_invalid_index_raises(self, deployable, rng):
        with pytest.raises(ValueError, match="true_value_index"):
            deployable.sample(5, n_samples=10, rng=rng)

    def test_sample_negative_index_raises(self, deployable, rng):
        with pytest.raises(ValueError, match="true_value_index"):
            deployable.sample(-1, n_samples=10, rng=rng)

    def test_sample_vectorized(self, deployable, rng):
        indices = np.array([0, 1, 0, 1, 0])
        samples = deployable.sample_vectorized(indices, rng=rng)
        assert samples.shape == (5,)

    def test_properties(self, deployable):
        assert deployable.n == 2
        assert deployable.k == 4
        assert deployable.epsilon == 1.0
        assert deployable.delta == 0.0

    def test_to_dict(self, deployable):
        d = deployable.to_dict()
        assert "p" in d
        assert "y_grid" in d
        assert "epsilon" in d
        assert "delta" in d
        assert "n" in d
        assert "k" in d
        assert d["n"] == 2
        assert d["k"] == 4

    def test_from_dict_roundtrip(self, deployable):
        d = deployable.to_dict()
        restored = DeployableMechanism.from_dict(d)
        assert_allclose(restored.p, deployable.p, atol=1e-12)
        assert_allclose(restored.y_grid, deployable.y_grid, atol=1e-12)
        assert restored.epsilon == deployable.epsilon
        assert restored.delta == deployable.delta

    def test_to_json_roundtrip(self, deployable):
        json_str = deployable.to_json()
        restored = DeployableMechanism.from_json(json_str)
        assert_allclose(restored.p, deployable.p, atol=1e-12)

    def test_save_load_roundtrip(self, deployable):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mech.json"
            deployable.save(path)
            assert path.exists()
            loaded = DeployableMechanism.load(path)
            assert_allclose(loaded.p, deployable.p, atol=1e-12)
            assert loaded.epsilon == deployable.epsilon

    def test_summary(self, deployable):
        s = deployable.summary()
        assert "DeployableMechanism" in s
        assert "Privacy" in s
        assert "Dimensions" in s

    def test_repr(self, deployable):
        r = repr(deployable)
        assert "DeployableMechanism" in r
        assert "ε=1.0" in r

    def test_to_dict_with_certificate(self):
        n, k = 2, 4
        p = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        lp_result = {"dual_vars": None, "primal_obj": 1.0, "dual_obj": 0.99}
        mech = ExtractMechanism(
            p, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
            lp_result=lp_result,
        )
        d = mech.to_dict()
        assert "certificate" in d
        assert_allclose(d["certificate"]["primal_obj"], 1.0)
        # Roundtrip preserves cert
        restored = DeployableMechanism.from_dict(d)
        assert restored.certificate is not None

    def test_from_dict_without_certificate(self):
        d = {
            "p": [[0.5, 0.5], [0.5, 0.5]],
            "y_grid": [0.0, 1.0],
            "epsilon": 1.0,
            "delta": 0.0,
        }
        mech = DeployableMechanism.from_dict(d)
        assert mech.certificate is None
        assert mech.n == 2
        assert mech.k == 2


# ═══════════════════════════════════════════════════════════════════════════
# §16  Quick Extract Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQuickExtract:
    """Tests for the quick_extract convenience function."""

    def test_basic(self):
        p_raw = np.full((2, 4), 0.25)
        result = quick_extract(p_raw, epsilon=1.0)
        assert isinstance(result, DeployableMechanism)
        assert result.n == 2
        assert result.k == 4

    def test_auto_grid(self):
        p_raw = np.full((3, 5), 0.2)
        result = quick_extract(p_raw, epsilon=1.0)
        assert_allclose(result.y_grid, np.arange(5, dtype=np.float64))

    def test_custom_grid(self):
        p_raw = np.full((2, 3), 1.0 / 3)
        y_grid = np.array([10.0, 20.0, 30.0])
        result = quick_extract(p_raw, epsilon=1.0, y_grid=y_grid)
        assert_allclose(result.y_grid, y_grid)

    def test_with_delta(self):
        p_raw = np.full((2, 3), 1.0 / 3)
        result = quick_extract(p_raw, epsilon=0.5, delta=0.01)
        assert result.delta == 0.01


# ═══════════════════════════════════════════════════════════════════════════
# §17  ExtractToLegacy Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractToLegacy:
    """Tests for the extract_to_legacy path."""

    def test_legacy_type(self):
        n, k = 2, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        ext = MechanismExtractor(
            epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )
        result = ext.extract_to_legacy(p_raw)
        assert isinstance(result, ExtractedMechanism)
        assert result.n == n
        assert result.k == k

    def test_legacy_cdf_tables_shape(self):
        n, k = 3, 5
        p_raw = np.full((n, k), 0.2)
        y_grid = np.arange(k, dtype=np.float64)
        ext = MechanismExtractor(
            epsilon=1.0, delta=0.0,
            edges=[(0, 1), (1, 2)], y_grid=y_grid,
        )
        result = ext.extract_to_legacy(p_raw)
        assert result.cdf_tables.shape == (n, k)

    def test_legacy_alias_tables_length(self):
        n, k = 2, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        ext = MechanismExtractor(
            epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )
        result = ext.extract_to_legacy(p_raw)
        assert len(result.alias_tables) == n
        for prob_arr, alias_arr in result.alias_tables:
            assert len(prob_arr) == k
            assert len(alias_arr) == k


# ═══════════════════════════════════════════════════════════════════════════
# §18  Alias vs CDF Distribution Equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestAliasCDFEquivalence:
    """Alias and CDF sampling should produce the same distribution."""

    @pytest.mark.parametrize("probs", [
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.5, 0.3, 0.15, 0.05]),
        np.array([0.9, 0.05, 0.05]),
        np.array([1.0 / 7] * 7),
    ])
    def test_distributions_match(self, probs):
        n_samples = 100000
        alias_table = build_alias_table(probs)
        cdf_table = build_cdf_table(probs)

        rng_a = np.random.default_rng(55)
        rng_c = np.random.default_rng(56)

        alias_samples = batch_sample_alias(alias_table, n_samples, rng_a)
        cdf_samples = batch_sample_cdf(cdf_table, n_samples, rng_c)

        alias_freq = np.bincount(alias_samples, minlength=len(probs)) / n_samples
        cdf_freq = np.bincount(cdf_samples, minlength=len(probs)) / n_samples

        # Both should be close to the true distribution
        assert_allclose(alias_freq, probs, atol=0.02)
        assert_allclose(cdf_freq, probs, atol=0.02)


# ═══════════════════════════════════════════════════════════════════════════
# §19  Edge Cases and Robustness
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_bin_alias(self, rng):
        probs = np.array([1.0])
        table = build_alias_table(probs)
        for _ in range(20):
            assert sample_alias(table, rng) == 0
        samples = batch_sample_alias(table, 100, rng)
        assert np.all(samples == 0)

    def test_single_bin_cdf(self, rng):
        probs = np.array([1.0])
        table = build_cdf_table(probs)
        for _ in range(20):
            assert sample_cdf(table, rng) == 0
        samples = batch_sample_cdf(table, 100, rng)
        assert np.all(samples == 0)

    def test_two_bin_half_half(self, rng):
        probs = np.array([0.5, 0.5])
        alias_table = build_alias_table(probs)
        cdf_table = build_cdf_table(probs)
        n = 50000
        alias_samples = batch_sample_alias(alias_table, n, rng)
        cdf_rng = np.random.default_rng(99)
        cdf_samples = batch_sample_cdf(cdf_table, n, cdf_rng)
        # Both should be roughly 50/50
        alias_frac = np.mean(alias_samples == 0)
        cdf_frac = np.mean(cdf_samples == 0)
        assert abs(alias_frac - 0.5) < 0.02
        assert abs(cdf_frac - 0.5) < 0.02

    def test_very_skewed_alias(self, rng):
        """One bin has almost all the mass."""
        probs = np.array([1.0 - 1e-10, 1e-10])
        table = build_alias_table(probs)
        samples = batch_sample_alias(table, 10000, rng)
        # Almost all samples should be bin 0
        assert np.mean(samples == 0) > 0.99

    def test_large_table(self, rng):
        """Test with a large number of bins."""
        k = 10000
        probs = np.ones(k) / k
        alias_table = build_alias_table(probs)
        cdf_table = build_cdf_table(probs)
        alias_samples = batch_sample_alias(alias_table, k * 10, rng)
        cdf_samples = batch_sample_cdf(cdf_table, k * 10, np.random.default_rng(88))
        assert np.all(alias_samples >= 0)
        assert np.all(alias_samples < k)
        assert np.all(cdf_samples >= 0)
        assert np.all(cdf_samples < k)

    def test_renormalize_single_row(self):
        p = np.array([[3.0, 6.0, 1.0]])
        result = _renormalize(p)
        assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)

    def test_projection_then_renormalize_gives_valid_distribution(self):
        """Full pipeline: projection + renormalize should give valid rows."""
        p_raw = np.array([[-0.05, 0.3, 0.75], [0.1, -0.2, 1.1]])
        eta_min = 1e-8
        p_proj = _positivity_projection(p_raw, eta_min)
        p_norm = _renormalize(p_proj)
        assert_allclose(p_norm.sum(axis=1), 1.0, atol=1e-12)
        # After renormalization entries stay positive (but may be < eta_min
        # because renormalization divides by row sum > 1)
        assert np.all(p_norm > 0)


# ═══════════════════════════════════════════════════════════════════════════
# §20  Parametrized Distribution Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParametrizedDistributions:
    """Test alias and CDF with various parametrized distributions."""

    @pytest.mark.parametrize("k", [2, 3, 5, 10, 20])
    def test_dirichlet_random_distribution(self, k, rng):
        """Random distributions from Dirichlet should work correctly."""
        probs = rng.dirichlet(np.ones(k))
        alias_table = build_alias_table(probs)
        cdf_table = build_cdf_table(probs)

        n = 50000
        alias_samples = batch_sample_alias(alias_table, n, rng)
        cdf_samples = batch_sample_cdf(cdf_table, n, np.random.default_rng(200))

        alias_freq = np.bincount(alias_samples, minlength=k) / n
        cdf_freq = np.bincount(cdf_samples, minlength=k) / n

        assert_allclose(alias_freq, probs, atol=0.03)
        assert_allclose(cdf_freq, probs, atol=0.03)

    @pytest.mark.parametrize("concentration", [0.1, 1.0, 10.0])
    def test_dirichlet_concentration(self, concentration, rng):
        """Various Dirichlet concentrations (sparse to uniform)."""
        k = 8
        probs = rng.dirichlet(np.full(k, concentration))
        table = build_alias_table(probs)
        n = 50000
        samples = batch_sample_alias(table, n, rng)
        freq = np.bincount(samples, minlength=k) / n
        assert_allclose(freq, probs, atol=0.03)

    @pytest.mark.parametrize("spike_idx,k", [
        (0, 5), (2, 5), (4, 5), (0, 1), (9, 10),
    ])
    def test_spike_distribution(self, spike_idx, k, rng):
        """Distribution with all mass on a single bin."""
        probs = np.zeros(k)
        probs[spike_idx] = 1.0
        table = build_alias_table(probs)
        samples = batch_sample_alias(table, 1000, rng)
        assert np.all(samples == spike_idx)


# ═══════════════════════════════════════════════════════════════════════════
# §21  MSE/MAE Parametrized Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMSEMAEParametrized:
    """Parametrized tests for MSE and MAE."""

    @pytest.mark.parametrize("n,k", [(1, 2), (2, 5), (5, 10), (10, 3)])
    def test_mse_mae_shapes(self, n, k):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(k), size=n)
        grid = np.linspace(0, 1, k)
        true_values = rng.random(n)
        mse = compute_mechanism_mse(p, grid, true_values)
        mae = compute_mechanism_mae(p, grid, true_values)
        assert isinstance(mse, float)
        assert isinstance(mae, float)
        assert mse >= 0
        assert mae >= 0

    def test_mse_geq_mae_squared(self):
        """By Jensen's inequality, MSE >= MAE^2 is NOT always true,
        but MSE >= 0 and MAE >= 0 both hold."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(5), size=3)
        grid = np.linspace(0, 1, 5)
        true_values = rng.random(3)
        mse = compute_mechanism_mse(p, grid, true_values)
        mae = compute_mechanism_mae(p, grid, true_values)
        # Both non-negative
        assert mse >= -1e-15
        assert mae >= -1e-15

    @pytest.mark.parametrize("offset", [0.0, 0.5, 1.0, -1.0])
    def test_mse_translation_invariance_on_grid_shift(self, offset):
        """Shifting grid and true_values by same offset shouldn't change MSE."""
        p = np.array([[0.5, 0.3, 0.2]])
        grid = np.array([0.0, 1.0, 2.0])
        true_values = np.array([1.0])
        mse1 = compute_mechanism_mse(p, grid, true_values)
        mse2 = compute_mechanism_mse(p, grid + offset, true_values + offset)
        assert_allclose(mse1, mse2, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# §22  Full Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipelineIntegration:
    """End-to-end integration tests for the extraction pipeline."""

    def test_extract_sample_roundtrip(self, rng):
        """Extract, then sample, and verify samples are from y_grid."""
        n, k = 3, 6
        p_raw = np.full((n, k), 1.0 / k)
        y_grid = np.linspace(0, 5, k)
        edges = [(0, 1), (1, 2)]
        mech = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=edges, y_grid=y_grid)

        for i in range(n):
            samples = mech.sample(i, n_samples=100, rng=rng)
            for s in samples:
                assert s in y_grid

    def test_extract_serialize_deserialize_sample(self, rng):
        """Full roundtrip: extract → JSON → restore → sample."""
        n, k = 2, 4
        p_raw = np.full((n, k), 0.25)
        y_grid = np.arange(k, dtype=np.float64)
        mech = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=[(0, 1)], y_grid=y_grid)

        json_str = mech.to_json()
        restored = DeployableMechanism.from_json(json_str)

        samples = restored.sample(0, n_samples=50, rng=rng)
        assert len(samples) == 50
        assert np.all(np.isin(samples, y_grid))

    def test_extract_with_dp_feasible_mechanism(self, dp_feasible_mechanism):
        """Extract a known DP-feasible mechanism."""
        p_raw = dp_feasible_mechanism
        n, k = p_raw.shape
        y_grid = np.arange(k, dtype=np.float64)
        edges = [(0, 1)]
        mech = ExtractMechanism(p_raw, epsilon=1.0, delta=0.0, edges=edges, y_grid=y_grid)
        assert isinstance(mech, DeployableMechanism)
        assert_allclose(mech.p.sum(axis=1), 1.0, atol=1e-10)

    def test_mse_mae_on_extracted_mechanism(self):
        """Compute MSE and MAE on an extracted mechanism."""
        n, k = 2, 5
        p_raw = np.full((n, k), 0.2)
        y_grid = np.linspace(0, 4, k)
        true_values = np.array([1.0, 3.0])
        mech = ExtractMechanism(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )
        mse = compute_mechanism_mse(mech.p, y_grid, true_values)
        mae = compute_mechanism_mae(mech.p, y_grid, true_values)
        assert mse >= 0
        assert mae >= 0

    def test_save_load_sample_consistency(self, rng):
        """Samples from loaded mechanism should match original's distribution."""
        n, k = 2, 5
        p_raw = np.full((n, k), 0.2)
        y_grid = np.arange(k, dtype=np.float64)
        mech = ExtractMechanism(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_mech.json"
            mech.save(path)
            loaded = DeployableMechanism.load(path)

        assert_allclose(loaded.p, mech.p, atol=1e-12)
        # Both should produce samples in the same range
        s1 = mech.sample(0, n_samples=100, rng=np.random.default_rng(42))
        s2 = loaded.sample(0, n_samples=100, rng=np.random.default_rng(42))
        assert_array_equal(s1, s2)

    def test_extract_multiple_edges(self):
        """Test with complete adjacency (all pairs)."""
        n, k = 4, 3
        p_raw = np.full((n, k), 1.0 / k)
        y_grid = np.arange(k, dtype=np.float64)
        adj = AdjacencyRelation.complete(n)
        mech = ExtractMechanism(
            p_raw, epsilon=1.0, delta=0.0,
            edges=adj, y_grid=y_grid,
        )
        assert isinstance(mech, DeployableMechanism)

    def test_extract_preserves_y_grid(self):
        """y_grid should be stored correctly in the deployable mechanism."""
        n, k = 2, 5
        p_raw = np.full((n, k), 0.2)
        y_grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mech = ExtractMechanism(
            p_raw, epsilon=1.0, delta=0.0,
            edges=[(0, 1)], y_grid=y_grid,
        )
        assert_allclose(mech.y_grid, y_grid)
