"""
Comprehensive tests for dp_forge.workloads — workload generation, analysis,
composition, benchmark workloads, and utility functions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dp_forge.exceptions import ConfigurationError
from dp_forge.types import QueryType, WorkloadSpec
from dp_forge.workloads import (
    T1_counting,
    T1_histogram_medium,
    T1_histogram_small,
    T1_prefix,
    T2_2d_histogram,
    T2_all_range,
    T2_histogram_large,
    T3_large_histogram,
    T3_marginals,
    WorkloadAnalyzer,
    WorkloadGenerator,
    WorkloadProperties,
    compose_workloads,
    subsampled_workload,
    summarize_workload,
    to_workload_spec,
    weighted_workload,
)


# ═══════════════════════════════════════════════════════════════════════════
# §1  WorkloadGenerator — identity
# ═══════════════════════════════════════════════════════════════════════════


class TestIdentity:
    """Tests for WorkloadGenerator.identity."""

    @pytest.mark.parametrize("d", [1, 2, 5, 10, 50])
    def test_shape(self, d: int):
        A = WorkloadGenerator.identity(d)
        assert A.shape == (d, d)

    @pytest.mark.parametrize("d", [1, 3, 8])
    def test_is_identity_matrix(self, d: int):
        A = WorkloadGenerator.identity(d)
        np.testing.assert_array_equal(A, np.eye(d))

    def test_dtype(self):
        A = WorkloadGenerator.identity(4)
        assert A.dtype == np.float64

    @pytest.mark.parametrize("d", [0, -1, -100])
    def test_invalid_d_raises(self, d: int):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.identity(d)

    def test_diagonal_values(self):
        A = WorkloadGenerator.identity(6)
        assert np.all(np.diag(A) == 1.0)

    def test_off_diagonal_zero(self):
        A = WorkloadGenerator.identity(6)
        off = A - np.diag(np.diag(A))
        assert np.all(off == 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# §2  WorkloadGenerator — prefix_sums
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefixSums:
    """Tests for WorkloadGenerator.prefix_sums."""

    @pytest.mark.parametrize("d", [1, 2, 5, 10])
    def test_shape(self, d: int):
        A = WorkloadGenerator.prefix_sums(d)
        assert A.shape == (d, d)

    def test_lower_triangular(self):
        d = 6
        A = WorkloadGenerator.prefix_sums(d)
        upper = np.triu(A, k=1)
        np.testing.assert_array_equal(upper, np.zeros_like(upper))

    def test_lower_entries_are_ones(self):
        d = 5
        A = WorkloadGenerator.prefix_sums(d)
        expected = np.tril(np.ones((d, d)))
        np.testing.assert_array_equal(A, expected)

    def test_first_row(self):
        A = WorkloadGenerator.prefix_sums(4)
        np.testing.assert_array_equal(A[0], [1, 0, 0, 0])

    def test_last_row_all_ones(self):
        d = 7
        A = WorkloadGenerator.prefix_sums(d)
        np.testing.assert_array_equal(A[-1], np.ones(d))

    def test_dtype(self):
        assert WorkloadGenerator.prefix_sums(3).dtype == np.float64

    @pytest.mark.parametrize("d", [0, -1])
    def test_invalid_d_raises(self, d: int):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.prefix_sums(d)

    def test_d_equals_one(self):
        A = WorkloadGenerator.prefix_sums(1)
        np.testing.assert_array_equal(A, [[1.0]])

    def test_prefix_sum_semantics(self):
        """Row i should sum entries x[0..i]."""
        d = 5
        A = WorkloadGenerator.prefix_sums(d)
        x = np.arange(1, d + 1, dtype=np.float64)
        result = A @ x
        expected = np.cumsum(x)
        np.testing.assert_array_almost_equal(result, expected)


# ═══════════════════════════════════════════════════════════════════════════
# §3  WorkloadGenerator — all_range
# ═══════════════════════════════════════════════════════════════════════════


class TestAllRange:
    """Tests for WorkloadGenerator.all_range (all contiguous range queries)."""

    @pytest.mark.parametrize("d", [1, 2, 3, 5, 8])
    def test_num_queries(self, d: int):
        A = WorkloadGenerator.all_range(d)
        expected_m = d * (d + 1) // 2
        assert A.shape == (expected_m, d)

    def test_d1_is_single_identity(self):
        A = WorkloadGenerator.all_range(1)
        np.testing.assert_array_equal(A, [[1.0]])

    def test_d2(self):
        A = WorkloadGenerator.all_range(2)
        assert A.shape == (3, 2)
        # Ranges: [0,0], [0,1], [1,1]
        assert np.sum(A) == 1 + 2 + 1  # 4 total ones

    def test_binary_entries(self):
        A = WorkloadGenerator.all_range(5)
        unique = np.unique(A)
        np.testing.assert_array_equal(unique, [0.0, 1.0])

    def test_contains_identity_rows(self):
        """Single-element ranges should appear (identity-like rows)."""
        d = 4
        A = WorkloadGenerator.all_range(d)
        identity = np.eye(d)
        for i in range(d):
            row = identity[i]
            found = any(np.allclose(A[r], row) for r in range(A.shape[0]))
            assert found, f"Identity row {i} not found in all_range"

    def test_total_range_row(self):
        """The full-range query [0, d-1] should be present (all ones)."""
        d = 5
        A = WorkloadGenerator.all_range(d)
        all_ones = np.ones(d)
        found = any(np.allclose(A[r], all_ones) for r in range(A.shape[0]))
        assert found

    @pytest.mark.parametrize("d", [0, -1])
    def test_invalid_d_raises(self, d: int):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.all_range(d)

    def test_each_row_is_contiguous(self):
        """Each row must be a contiguous block of 1s."""
        d = 6
        A = WorkloadGenerator.all_range(d)
        for r in range(A.shape[0]):
            row = A[r]
            nz = np.where(row > 0.5)[0]
            assert len(nz) > 0
            # Check contiguity
            assert np.all(np.diff(nz) == 1), f"Row {r} is not contiguous"

    def test_d3_explicit(self):
        A = WorkloadGenerator.all_range(3)
        assert A.shape == (6, 3)


# ═══════════════════════════════════════════════════════════════════════════
# §4  WorkloadGenerator — histogram_1d, histogram_2d
# ═══════════════════════════════════════════════════════════════════════════


class TestHistogram:
    """Tests for histogram workloads."""

    @pytest.mark.parametrize("n", [1, 5, 10, 20])
    def test_histogram_1d_shape(self, n: int):
        A = WorkloadGenerator.histogram_1d(n)
        assert A.shape == (n, n)

    @pytest.mark.parametrize("n", [3, 7])
    def test_histogram_1d_is_identity(self, n: int):
        A = WorkloadGenerator.histogram_1d(n)
        np.testing.assert_array_equal(A, np.eye(n))

    def test_histogram_1d_invalid(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.histogram_1d(0)

    @pytest.mark.parametrize("n1,n2", [(2, 3), (5, 5), (1, 10), (4, 7)])
    def test_histogram_2d_shape(self, n1: int, n2: int):
        d = n1 * n2
        A = WorkloadGenerator.histogram_2d(n1, n2)
        assert A.shape == (d, d)

    def test_histogram_2d_is_identity(self):
        A = WorkloadGenerator.histogram_2d(3, 4)
        np.testing.assert_array_equal(A, np.eye(12))

    def test_histogram_2d_invalid_n1(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.histogram_2d(0, 5)

    def test_histogram_2d_invalid_n2(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.histogram_2d(5, 0)

    def test_histogram_2d_both_invalid(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.histogram_2d(-1, -1)


# ═══════════════════════════════════════════════════════════════════════════
# §5  WorkloadGenerator — marginals
# ═══════════════════════════════════════════════════════════════════════════


class TestMarginals:
    """Tests for WorkloadGenerator.marginals."""

    def test_shape_d3_k1(self):
        A = WorkloadGenerator.marginals(3, 1)
        # C(3,1)=3 marginals, 2^1=2 cells, 2^3=8 domain
        assert A.shape == (6, 8)

    def test_shape_d3_k2(self):
        A = WorkloadGenerator.marginals(3, 2)
        # C(3,2)=3 marginals, 2^2=4 cells, 2^3=8 domain
        assert A.shape == (12, 8)

    def test_shape_d4_k2(self):
        A = WorkloadGenerator.marginals(4, 2)
        # C(4,2)=6 marginals, 2^2=4 cells, 2^4=16 domain
        assert A.shape == (24, 16)

    def test_shape_d3_k3(self):
        A = WorkloadGenerator.marginals(3, 3)
        # C(3,3)=1, 2^3=8 cells, 2^3=8 domain → full identity-like
        assert A.shape == (8, 8)

    def test_binary_entries(self):
        A = WorkloadGenerator.marginals(3, 2)
        unique = np.unique(A)
        assert set(unique) <= {0.0, 1.0}

    def test_rows_sum_to_power_of_two(self):
        """Each marginal cell query should select 2^(d-k) domain elements."""
        d, k = 4, 2
        A = WorkloadGenerator.marginals(d, k)
        expected_sum = 2 ** (d - k)
        for r in range(A.shape[0]):
            assert np.sum(A[r]) == expected_sum

    def test_invalid_k_zero(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.marginals(3, 0)

    def test_invalid_k_exceeds_d(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.marginals(3, 4)

    def test_invalid_d_zero(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.marginals(0, 1)

    def test_d_too_large(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.marginals(17, 1)

    def test_k_equals_d(self):
        """When k=d, the marginal is the full histogram (identity permutation)."""
        d = 3
        A = WorkloadGenerator.marginals(d, d)
        assert A.shape == (2**d, 2**d)
        # Each row should select exactly 1 domain element
        for r in range(A.shape[0]):
            assert np.sum(A[r]) == 1.0

    def test_column_sums(self):
        """Each domain element should appear in the same total count of queries."""
        d, k = 3, 1
        A = WorkloadGenerator.marginals(d, k)
        col_sums = np.sum(A, axis=0)
        # By symmetry, all column sums should be equal
        assert np.all(col_sums == col_sums[0])


# ═══════════════════════════════════════════════════════════════════════════
# §6  WorkloadGenerator — random_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestRandomWorkload:
    """Tests for WorkloadGenerator.random_workload."""

    def test_shape(self):
        A = WorkloadGenerator.random_workload(5, 10, seed=42)
        assert A.shape == (10, 5)

    def test_reproducibility(self):
        A1 = WorkloadGenerator.random_workload(5, 10, seed=42)
        A2 = WorkloadGenerator.random_workload(5, 10, seed=42)
        np.testing.assert_array_equal(A1, A2)

    def test_different_seeds_differ(self):
        A1 = WorkloadGenerator.random_workload(5, 10, seed=1)
        A2 = WorkloadGenerator.random_workload(5, 10, seed=2)
        assert not np.allclose(A1, A2)

    def test_density_full(self):
        A = WorkloadGenerator.random_workload(10, 20, density=1.0, seed=0)
        # With density=1.0, all entries should generally be non-zero
        # (though Gaussian can rarely produce exact 0)
        assert A.shape == (20, 10)

    def test_density_sparse(self):
        A = WorkloadGenerator.random_workload(100, 200, density=0.1, seed=42)
        nnz_frac = np.count_nonzero(A) / A.size
        # Should be roughly 10%, allow tolerance
        assert nnz_frac < 0.25

    def test_dtype(self):
        A = WorkloadGenerator.random_workload(5, 10, seed=0)
        assert A.dtype == np.float64

    @pytest.mark.parametrize("bad_d", [0, -1])
    def test_invalid_d(self, bad_d: int):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.random_workload(bad_d, 5, seed=0)

    @pytest.mark.parametrize("bad_m", [0, -1])
    def test_invalid_m(self, bad_m: int):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.random_workload(5, bad_m, seed=0)

    def test_invalid_density_zero(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.random_workload(5, 10, density=0.0, seed=0)

    def test_invalid_density_negative(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.random_workload(5, 10, density=-0.5, seed=0)

    def test_scale_parameter(self):
        A_small = WorkloadGenerator.random_workload(10, 20, scale=0.01, seed=42)
        A_large = WorkloadGenerator.random_workload(10, 20, scale=100.0, seed=42)
        assert np.std(A_large) > np.std(A_small) * 10


# ═══════════════════════════════════════════════════════════════════════════
# §7  WorkloadGenerator — custom_linear
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomLinear:
    """Tests for WorkloadGenerator.custom_linear."""

    def test_basic(self):
        M = np.array([[1, 0], [1, 1]])
        A = WorkloadGenerator.custom_linear(M)
        np.testing.assert_array_equal(A, M)
        assert A.dtype == np.float64

    def test_returns_copy(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        A = WorkloadGenerator.custom_linear(M)
        A[0, 0] = 999.0
        assert M[0, 0] == 1.0

    def test_1d_input_reshaped(self):
        v = np.array([1.0, 2.0, 3.0])
        A = WorkloadGenerator.custom_linear(v)
        assert A.shape == (1, 3)

    def test_non_finite_raises(self):
        M = np.array([[1.0, np.inf], [0.0, 1.0]])
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.custom_linear(M)

    def test_nan_raises(self):
        M = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.custom_linear(M)


# ═══════════════════════════════════════════════════════════════════════════
# §8  WorkloadGenerator — wavelet_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestWaveletWorkload:
    """Tests for WorkloadGenerator.wavelet_workload."""

    @pytest.mark.parametrize("d", [1, 2, 4, 8, 16])
    def test_shape(self, d: int):
        W = WorkloadGenerator.wavelet_workload(d)
        assert W.shape == (d, d)

    def test_not_power_of_two_raises(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.wavelet_workload(3)

    def test_d_zero_raises(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.wavelet_workload(0)

    def test_d1(self):
        W = WorkloadGenerator.wavelet_workload(1)
        np.testing.assert_array_almost_equal(W, [[1.0]])

    def test_orthogonal_rows_d4(self):
        """Wavelet rows should be orthogonal."""
        W = WorkloadGenerator.wavelet_workload(4)
        gram = W @ W.T
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# §9  WorkloadGenerator — fourier_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestFourierWorkload:
    """Tests for WorkloadGenerator.fourier_workload."""

    @pytest.mark.parametrize("d", [1, 4, 8])
    def test_shape(self, d: int):
        F = WorkloadGenerator.fourier_workload(d)
        assert F.shape == (d, d)

    def test_dtype(self):
        F = WorkloadGenerator.fourier_workload(4)
        assert F.dtype == np.float64

    def test_d_zero_raises(self):
        with pytest.raises(ConfigurationError):
            WorkloadGenerator.fourier_workload(0)

    def test_first_row_constant(self):
        """First row of DFT is the constant (DC) component."""
        d = 8
        F = WorkloadGenerator.fourier_workload(d)
        # First row should be all equal (DC component)
        assert np.allclose(F[0], F[0, 0])


# ═══════════════════════════════════════════════════════════════════════════
# §10  WorkloadAnalyzer — analyze_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkloadAnalyzer:
    """Tests for WorkloadAnalyzer.analyze_workload."""

    def setup_method(self):
        self.analyzer = WorkloadAnalyzer()

    # --- Rank ---

    @pytest.mark.parametrize("d", [1, 3, 5, 10])
    def test_identity_rank(self, d: int):
        A = WorkloadGenerator.identity(d)
        props = self.analyzer.analyze_workload(A)
        assert props.rank == d

    @pytest.mark.parametrize("d", [2, 5, 8])
    def test_prefix_rank(self, d: int):
        A = WorkloadGenerator.prefix_sums(d)
        props = self.analyzer.analyze_workload(A)
        assert props.rank == d

    def test_rank_deficient(self):
        A = np.array([[1, 0], [2, 0], [0, 1]])
        props = self.analyzer.analyze_workload(A.astype(np.float64))
        assert props.rank == 2

    def test_zero_matrix_rank(self):
        A = np.zeros((3, 3))
        props = self.analyzer.analyze_workload(A)
        assert props.rank == 0

    # --- Condition number ---

    @pytest.mark.parametrize("d", [2, 5, 10])
    def test_identity_condition_number(self, d: int):
        A = WorkloadGenerator.identity(d)
        props = self.analyzer.analyze_workload(A)
        assert abs(props.condition_number - 1.0) < 1e-10

    def test_prefix_condition_number_greater_than_one(self):
        A = WorkloadGenerator.prefix_sums(5)
        props = self.analyzer.analyze_workload(A)
        assert props.condition_number > 1.0

    def test_ill_conditioned_matrix(self):
        A = np.array([[1, 0], [0, 1e-6]])
        props = self.analyzer.analyze_workload(A)
        assert props.condition_number > 1e5

    # --- Sparsity ---

    def test_identity_sparsity(self):
        d = 10
        A = WorkloadGenerator.identity(d)
        props = self.analyzer.analyze_workload(A)
        expected = 1.0 - d / (d * d)  # = 1 - 1/d = 0.9
        assert abs(props.sparsity - expected) < 1e-10

    def test_full_matrix_sparsity(self):
        A = np.ones((3, 3))
        props = self.analyzer.analyze_workload(A)
        assert abs(props.sparsity) < 1e-10

    def test_zero_matrix_sparsity(self):
        A = np.zeros((4, 4))
        props = self.analyzer.analyze_workload(A)
        assert abs(props.sparsity - 1.0) < 1e-10

    # --- Sensitivity ---

    def test_identity_sensitivity_l1(self):
        A = WorkloadGenerator.identity(5)
        props = self.analyzer.analyze_workload(A)
        assert abs(props.sensitivity_l1 - 1.0) < 1e-10

    def test_identity_sensitivity_l2(self):
        A = WorkloadGenerator.identity(5)
        props = self.analyzer.analyze_workload(A)
        assert abs(props.sensitivity_l2 - 1.0) < 1e-10

    def test_prefix_sensitivity_l1(self):
        d = 5
        A = WorkloadGenerator.prefix_sums(d)
        props = self.analyzer.analyze_workload(A)
        # First column has d ones → L1 sensitivity = d
        assert abs(props.sensitivity_l1 - d) < 1e-10

    # --- Shape properties ---

    def test_is_square_identity(self):
        props = self.analyzer.analyze_workload(np.eye(5))
        assert props.is_square is True

    def test_is_not_square(self):
        A = np.ones((3, 5))
        props = self.analyzer.analyze_workload(A)
        assert props.is_square is False

    def test_is_full_rank_identity(self):
        props = self.analyzer.analyze_workload(np.eye(5))
        assert props.is_full_rank is True

    def test_is_orthogonal_identity(self):
        props = self.analyzer.analyze_workload(np.eye(5))
        assert props.is_orthogonal is True

    def test_prefix_not_orthogonal(self):
        A = WorkloadGenerator.prefix_sums(5)
        props = self.analyzer.analyze_workload(A)
        assert props.is_orthogonal is False

    # --- 1D input ---

    def test_1d_input_is_handled(self):
        v = np.array([1.0, 2.0, 3.0])
        props = self.analyzer.analyze_workload(v)
        assert props.shape == (1, 3)

    # --- Invalid input ---

    def test_non_finite_raises(self):
        A = np.array([[1.0, np.inf], [0.0, 1.0]])
        with pytest.raises(ConfigurationError):
            self.analyzer.analyze_workload(A)


# ═══════════════════════════════════════════════════════════════════════════
# §11  WorkloadAnalyzer — detect_structure
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectStructure:
    """Tests for WorkloadAnalyzer.detect_structure."""

    def setup_method(self):
        self.analyzer = WorkloadAnalyzer()

    def test_identity_detected(self):
        A = np.eye(5)
        s = self.analyzer.detect_structure(A)
        assert s["identity"] is True

    def test_prefix_lower_triangular(self):
        A = WorkloadGenerator.prefix_sums(4)
        s = self.analyzer.detect_structure(A)
        assert s["lower_triangular"] is True
        assert s["binary"] is True
        assert s["non_negative"] is True

    def test_prefix_not_identity(self):
        A = WorkloadGenerator.prefix_sums(4)
        s = self.analyzer.detect_structure(A)
        assert s["identity"] is False

    def test_toeplitz_identity(self):
        A = np.eye(4)
        s = self.analyzer.detect_structure(A)
        assert s["toeplitz"] is True

    def test_circulant_detected(self):
        row = [1, 2, 3, 4]
        d = len(row)
        A = np.zeros((d, d))
        for i in range(d):
            A[i] = np.roll(row, i)
        s = self.analyzer.detect_structure(A)
        assert s["circulant"] is True

    def test_non_square_not_lower_triangular(self):
        A = np.ones((3, 5))
        s = self.analyzer.detect_structure(A)
        assert s["lower_triangular"] is False
        assert s["upper_triangular"] is False

    def test_block_diagonal(self):
        A = np.zeros((4, 4))
        A[0:2, 0:2] = np.eye(2)
        A[2:4, 2:4] = np.eye(2)
        s = self.analyzer.detect_structure(A)
        assert s["block_diagonal"] is True

    def test_symmetric_detected(self):
        A = np.array([[1.0, 2.0], [2.0, 3.0]])
        s = self.analyzer.detect_structure(A)
        assert s["symmetric"] is True

    def test_non_symmetric_detected(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = self.analyzer.detect_structure(A)
        assert s["symmetric"] is False


# ═══════════════════════════════════════════════════════════════════════════
# §12  WorkloadAnalyzer — optimal_strategy_bounds
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimalStrategyBounds:
    """Tests for WorkloadAnalyzer.optimal_strategy_bounds."""

    def setup_method(self):
        self.analyzer = WorkloadAnalyzer()

    def test_identity_bounds(self):
        A = np.eye(3)
        bounds = self.analyzer.optimal_strategy_bounds(A, 1.0)
        assert "svd_lower" in bounds
        assert "identity_upper" in bounds
        assert "sensitivity_upper" in bounds
        assert bounds["svd_lower"] <= bounds["identity_upper"]

    def test_identity_upper_value(self):
        d = 3
        A = np.eye(d)
        bounds = self.analyzer.optimal_strategy_bounds(A, 1.0)
        # trace(A A^T) = d, identity_upper = 2*d/eps^2 = 2*3 = 6
        assert abs(bounds["identity_upper"] - 2 * d) < 1e-10

    def test_epsilon_stored(self):
        bounds = self.analyzer.optimal_strategy_bounds(np.eye(3), 0.5)
        assert bounds["epsilon"] == 0.5

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ConfigurationError):
            self.analyzer.optimal_strategy_bounds(np.eye(3), 0.0)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ConfigurationError):
            self.analyzer.optimal_strategy_bounds(np.eye(3), -1.0)

    def test_lower_leq_upper(self):
        A = WorkloadGenerator.prefix_sums(5)
        bounds = self.analyzer.optimal_strategy_bounds(A, 1.0)
        assert bounds["svd_lower"] <= bounds["identity_upper"] + 1e-10

    def test_max_singular_value(self):
        A = np.eye(3)
        bounds = self.analyzer.optimal_strategy_bounds(A, 1.0)
        assert abs(bounds["max_singular_value"] - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# §13  WorkloadAnalyzer — workload_factorization
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkloadFactorization:
    """Tests for WorkloadAnalyzer.workload_factorization."""

    def setup_method(self):
        self.analyzer = WorkloadAnalyzer()

    def test_identity_factorization(self):
        A = np.eye(4)
        fac = self.analyzer.workload_factorization(A)
        assert fac["rank"] == 4
        assert fac["approx_error"] < 1e-10

    def test_low_rank_approx(self):
        A = np.eye(5)
        fac = self.analyzer.workload_factorization(A, rank=2)
        assert fac["target_rank"] == 2
        # Low-rank approximation should have some error
        recon = fac["U"] @ np.diag(fac["S"]) @ fac["Vt"]
        assert recon.shape == (5, 5)

    def test_non_negative_flag(self):
        A = np.eye(3)
        fac = self.analyzer.workload_factorization(A)
        assert fac["is_non_negative"] is True

        B = np.array([[1.0, -1.0], [0.0, 1.0]])
        fac2 = self.analyzer.workload_factorization(B)
        assert fac2["is_non_negative"] is False


# ═══════════════════════════════════════════════════════════════════════════
# §14  WorkloadAnalyzer — sensitivity ball vertices
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityBallVertices:
    """Tests for WorkloadAnalyzer.compute_sensitivity_ball_vertices."""

    def setup_method(self):
        self.analyzer = WorkloadAnalyzer()

    def test_l1_identity(self):
        A = np.eye(3)
        v = self.analyzer.compute_sensitivity_ball_vertices(A, norm_ord=1)
        assert v.shape == (6, 3)

    def test_l2_identity(self):
        A = np.eye(3)
        v = self.analyzer.compute_sensitivity_ball_vertices(A, norm_ord=2)
        # Each column has unit norm, so 2*3=6 vertices
        assert v.shape[0] == 6

    def test_vertices_include_columns(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        v = self.analyzer.compute_sensitivity_ball_vertices(A, norm_ord=1)
        # Should include columns of A
        col0 = A[:, 0]
        found = any(np.allclose(v[r], col0) for r in range(v.shape[0]))
        assert found


# ═══════════════════════════════════════════════════════════════════════════
# §15  WorkloadProperties dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkloadProperties:
    """Tests for the WorkloadProperties dataclass."""

    def test_repr(self):
        props = WorkloadProperties(
            shape=(5, 5),
            rank=5,
            condition_number=1.0,
            sparsity=0.8,
            sensitivity_l1=1.0,
            sensitivity_l2=1.0,
            sensitivity_linf=1.0,
            is_square=True,
            is_orthogonal=True,
            is_full_rank=True,
            structure={"identity": True, "toeplitz": True},
        )
        r = repr(props)
        assert "WorkloadProperties" in r
        assert "identity" in r
        assert "toeplitz" in r

    def test_repr_no_structures(self):
        props = WorkloadProperties(
            shape=(3, 5),
            rank=3,
            condition_number=2.0,
            sparsity=0.5,
            sensitivity_l1=1.0,
            sensitivity_l2=1.0,
            sensitivity_linf=1.0,
            is_square=False,
            is_orthogonal=False,
            is_full_rank=True,
            structure={},
        )
        r = repr(props)
        assert "none" in r


# ═══════════════════════════════════════════════════════════════════════════
# §16  compose_workloads
# ═══════════════════════════════════════════════════════════════════════════


class TestComposeWorkloads:
    """Tests for compose_workloads (vertical stacking)."""

    def test_basic_stack(self):
        A1 = np.eye(3)
        A2 = np.ones((2, 3))
        C = compose_workloads(A1, A2)
        assert C.shape == (5, 3)

    def test_content_preserved(self):
        A1 = np.eye(3)
        A2 = 2 * np.ones((1, 3))
        C = compose_workloads(A1, A2)
        np.testing.assert_array_equal(C[:3], A1)
        np.testing.assert_array_equal(C[3:], A2)

    def test_single_workload(self):
        A = np.eye(4)
        C = compose_workloads(A)
        np.testing.assert_array_equal(C, A)

    def test_three_workloads(self):
        A1 = np.eye(2)
        A2 = np.ones((3, 2))
        A3 = np.zeros((1, 2))
        C = compose_workloads(A1, A2, A3)
        assert C.shape == (6, 2)

    def test_incompatible_columns_raises(self):
        A1 = np.eye(3)
        A2 = np.eye(4)
        with pytest.raises(ConfigurationError):
            compose_workloads(A1, A2)

    def test_no_workloads_raises(self):
        with pytest.raises(ConfigurationError):
            compose_workloads()

    def test_1d_input_handled(self):
        v = np.array([1.0, 2.0, 3.0])
        A = np.eye(3)
        C = compose_workloads(v, A)
        assert C.shape == (4, 3)


# ═══════════════════════════════════════════════════════════════════════════
# §17  weighted_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestWeightedWorkload:
    """Tests for weighted_workload."""

    def test_basic(self):
        A = np.eye(3)
        w = np.array([1.0, 2.0, 3.0])
        Aw = weighted_workload(A, w)
        expected = np.diag([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(Aw, expected)

    def test_shape_preserved(self):
        A = np.ones((4, 5))
        w = np.array([1.0, 2.0, 3.0, 4.0])
        Aw = weighted_workload(A, w)
        assert Aw.shape == (4, 5)

    def test_negative_weight_raises(self):
        A = np.eye(2)
        w = np.array([1.0, -1.0])
        with pytest.raises(ConfigurationError):
            weighted_workload(A, w)

    def test_mismatched_length_raises(self):
        A = np.eye(3)
        w = np.array([1.0, 2.0])
        with pytest.raises(ConfigurationError):
            weighted_workload(A, w)

    def test_zero_weight(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([0.0, 1.0])
        Aw = weighted_workload(A, w)
        np.testing.assert_array_equal(Aw[0], [0.0, 0.0])
        np.testing.assert_array_equal(Aw[1], [3.0, 4.0])


# ═══════════════════════════════════════════════════════════════════════════
# §18  subsampled_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestSubsampledWorkload:
    """Tests for subsampled_workload."""

    def test_columns_preserved(self):
        A = np.eye(10)
        S = subsampled_workload(A, 0.5, seed=42)
        assert S.shape[1] == 10

    def test_fewer_rows(self):
        A = np.eye(100)
        S = subsampled_workload(A, 0.3, seed=42)
        assert S.shape[0] < 100

    def test_rate_one(self):
        A = np.eye(5)
        S = subsampled_workload(A, 1.0, seed=0)
        assert S.shape[0] == 5

    def test_invalid_rate_zero(self):
        with pytest.raises(ConfigurationError):
            subsampled_workload(np.eye(5), 0.0)

    def test_invalid_rate_negative(self):
        with pytest.raises(ConfigurationError):
            subsampled_workload(np.eye(5), -0.1)

    def test_at_least_one_row(self):
        """Even with very low rate, should return at least 1 row."""
        A = np.eye(5)
        S = subsampled_workload(A, 0.01, seed=99)
        assert S.shape[0] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# §19  Benchmark workloads — Tier 1
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkT1:
    """Tests for Tier 1 benchmark workloads."""

    def test_counting_shape(self):
        A = T1_counting(5)
        assert A.shape == (1, 5)

    def test_counting_single_one(self):
        A = T1_counting(5)
        assert np.sum(A) == 1.0
        assert A[0, 0] == 1.0

    def test_counting_default(self):
        A = T1_counting()
        assert A.shape == (1, 5)

    def test_counting_invalid(self):
        with pytest.raises(ConfigurationError):
            T1_counting(0)

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_counting_custom_n(self, d: int):
        A = T1_counting(d)
        assert A.shape == (1, d)

    def test_histogram_small_shape(self):
        A = T1_histogram_small()
        assert A.shape == (5, 5)

    def test_histogram_small_is_identity(self):
        A = T1_histogram_small()
        np.testing.assert_array_equal(A, np.eye(5))

    def test_histogram_small_custom_d(self):
        A = T1_histogram_small(d=8)
        assert A.shape == (8, 8)

    def test_histogram_medium_shape(self):
        A = T1_histogram_medium()
        assert A.shape == (10, 10)

    def test_histogram_medium_is_identity(self):
        np.testing.assert_array_equal(T1_histogram_medium(), np.eye(10))

    def test_prefix_shape(self):
        A = T1_prefix()
        assert A.shape == (10, 10)

    def test_prefix_is_lower_triangular(self):
        A = T1_prefix()
        expected = np.tril(np.ones((10, 10)))
        np.testing.assert_array_equal(A, expected)

    def test_prefix_custom_d(self):
        A = T1_prefix(d=4)
        assert A.shape == (4, 4)


# ═══════════════════════════════════════════════════════════════════════════
# §20  Benchmark workloads — Tier 2
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkT2:
    """Tests for Tier 2 benchmark workloads."""

    def test_histogram_large_shape(self):
        A = T2_histogram_large()
        assert A.shape == (20, 20)

    def test_histogram_large_is_identity(self):
        np.testing.assert_array_equal(T2_histogram_large(), np.eye(20))

    def test_histogram_large_custom(self):
        A = T2_histogram_large(d=15)
        assert A.shape == (15, 15)

    def test_all_range_shape(self):
        A = T2_all_range()
        assert A.shape == (55, 10)

    def test_all_range_custom(self):
        A = T2_all_range(d=5)
        assert A.shape == (15, 5)

    def test_2d_histogram_shape(self):
        A = T2_2d_histogram()
        assert A.shape == (25, 25)

    def test_2d_histogram_is_identity(self):
        np.testing.assert_array_equal(T2_2d_histogram(), np.eye(25))

    def test_2d_histogram_custom(self):
        A = T2_2d_histogram(n1=3, n2=4)
        assert A.shape == (12, 12)


# ═══════════════════════════════════════════════════════════════════════════
# §21  Benchmark workloads — Tier 3
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkT3:
    """Tests for Tier 3 benchmark workloads."""

    def test_large_histogram_shape(self):
        A = T3_large_histogram()
        assert A.shape == (50, 50)

    def test_large_histogram_is_identity(self):
        np.testing.assert_array_equal(T3_large_histogram(), np.eye(50))

    def test_large_histogram_custom(self):
        A = T3_large_histogram(d=30)
        assert A.shape == (30, 30)

    def test_marginals_shape(self):
        A = T3_marginals()
        # d=10, k=3 → C(10,3)*2^3 = 120*8 = 960 rows, 2^10=1024 cols
        assert A.shape == (960, 1024)

    def test_marginals_custom(self):
        A = T3_marginals(d=4, k=2)
        # C(4,2)*2^2 = 6*4 = 24 rows, 2^4 = 16 cols
        assert A.shape == (24, 16)

    def test_marginals_binary(self):
        A = T3_marginals(d=4, k=2)
        unique = np.unique(A)
        assert set(unique) <= {0.0, 1.0}


# ═══════════════════════════════════════════════════════════════════════════
# §22  to_workload_spec
# ═══════════════════════════════════════════════════════════════════════════


class TestToWorkloadSpec:
    """Tests for to_workload_spec conversion."""

    def test_returns_workload_spec(self):
        A = np.eye(3)
        spec = to_workload_spec(A)
        assert isinstance(spec, WorkloadSpec)

    def test_matrix_stored(self):
        A = np.eye(4)
        spec = to_workload_spec(A)
        np.testing.assert_array_equal(spec.matrix, A)

    def test_default_query_type(self):
        spec = to_workload_spec(np.eye(3))
        assert spec.query_type == QueryType.LINEAR_WORKLOAD

    def test_custom_query_type(self):
        spec = to_workload_spec(np.eye(3), query_type=QueryType.HISTOGRAM)
        assert spec.query_type == QueryType.HISTOGRAM

    def test_auto_detect_identity(self):
        spec = to_workload_spec(np.eye(5))
        assert spec.structural_hint == "identity"

    def test_auto_detect_prefix_toeplitz(self):
        A = WorkloadGenerator.prefix_sums(5)
        spec = to_workload_spec(A)
        assert spec.structural_hint == "toeplitz"

    def test_manual_hint_overrides(self):
        A = np.eye(5)
        spec = to_workload_spec(A, structural_hint="custom")
        assert spec.structural_hint == "custom"

    def test_no_auto_detect(self):
        A = np.eye(5)
        spec = to_workload_spec(A, auto_detect=False)
        assert spec.structural_hint is None

    def test_circulant_detected(self):
        row = [1.0, 2.0, 3.0, 4.0]
        d = len(row)
        A = np.zeros((d, d))
        for i in range(d):
            A[i] = np.roll(row, i)
        spec = to_workload_spec(A)
        assert spec.structural_hint == "circulant"


# ═══════════════════════════════════════════════════════════════════════════
# §23  summarize_workload
# ═══════════════════════════════════════════════════════════════════════════


class TestSummarizeWorkload:
    """Tests for summarize_workload."""

    def test_returns_string(self):
        s = summarize_workload(np.eye(3))
        assert isinstance(s, str)

    def test_contains_header(self):
        s = summarize_workload(np.eye(3))
        assert "Workload Summary" in s

    def test_contains_shape(self):
        s = summarize_workload(np.eye(5))
        assert "5 queries" in s
        assert "5 domain" in s

    def test_contains_rank(self):
        s = summarize_workload(np.eye(4))
        assert "Rank:" in s

    def test_contains_sensitivity(self):
        s = summarize_workload(np.eye(3))
        assert "L1:" in s
        assert "L2:" in s
        assert "Linf:" in s

    def test_contains_structure(self):
        s = summarize_workload(np.eye(3))
        assert "Structure:" in s

    def test_prefix_summary(self):
        A = WorkloadGenerator.prefix_sums(5)
        s = summarize_workload(A)
        assert "Workload Summary" in s
        assert "5 queries" in s


# ═══════════════════════════════════════════════════════════════════════════
# §24  Cross-cutting / integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_compose_then_analyze(self):
        A1 = WorkloadGenerator.identity(5)
        A2 = WorkloadGenerator.prefix_sums(5)
        C = compose_workloads(A1, A2)
        analyzer = WorkloadAnalyzer()
        props = analyzer.analyze_workload(C)
        assert props.shape == (10, 5)
        assert props.rank == 5

    def test_random_workload_analysis(self):
        A = WorkloadGenerator.random_workload(10, 20, seed=42)
        analyzer = WorkloadAnalyzer()
        props = analyzer.analyze_workload(A)
        assert props.shape == (20, 10)
        assert props.rank <= 10

    def test_all_range_full_rank(self):
        d = 5
        A = WorkloadGenerator.all_range(d)
        analyzer = WorkloadAnalyzer()
        props = analyzer.analyze_workload(A)
        assert props.rank == d
        assert props.is_full_rank is True

    def test_benchmark_to_spec_roundtrip(self):
        A = T1_prefix()
        spec = to_workload_spec(A)
        assert isinstance(spec, WorkloadSpec)
        np.testing.assert_array_equal(spec.matrix, A)

    def test_weighted_then_compose(self):
        A = np.eye(3)
        w = np.array([1.0, 2.0, 3.0])
        Aw = weighted_workload(A, w)
        C = compose_workloads(A, Aw)
        assert C.shape == (6, 3)

    def test_analyze_all_benchmark_workloads(self):
        """All benchmark workloads should be analyzable without error."""
        analyzer = WorkloadAnalyzer()
        benchmarks = [
            T1_counting(),
            T1_histogram_small(),
            T1_histogram_medium(),
            T1_prefix(),
            T2_histogram_large(),
            T2_all_range(),
            T2_2d_histogram(),
            T3_large_histogram(),
        ]
        for A in benchmarks:
            props = analyzer.analyze_workload(A)
            assert props.rank >= 1
            assert props.condition_number >= 1.0

    def test_summarize_all_benchmarks(self):
        """summarize_workload should work on all benchmark workloads."""
        benchmarks = [
            T1_counting(),
            T1_histogram_small(),
            T1_prefix(),
            T2_all_range(),
        ]
        for A in benchmarks:
            s = summarize_workload(A)
            assert "Workload Summary" in s
            assert len(s) > 50
