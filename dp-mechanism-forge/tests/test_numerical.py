"""Comprehensive tests for dp_forge.numerical utilities.

Covers sparse matrix operations, numerical stability primitives,
tolerance management, condition number analysis, matrix projections,
stochastic checks, divergence measures, and error metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from dp_forge.exceptions import ConfigurationError, NumericalInstabilityError
from dp_forge.numerical import (
    adaptive_tolerance,
    build_csr,
    build_csr_from_dense,
    check_condition_and_raise,
    check_tolerance_consistency,
    compute_dp_tolerance,
    compute_output_grid,
    detect_near_singularity,
    diagonal_preconditioning,
    entropy,
    estimate_condition,
    frobenius_norm,
    is_doubly_stochastic,
    is_psd,
    is_stochastic,
    is_symmetric,
    kl_divergence,
    linf_error,
    log_subtract_exp,
    log_sum_exp,
    log_sum_exp_array,
    matrix_rank_estimate,
    normalize_rows,
    privacy_loss_rv,
    project_psd,
    project_simplex,
    project_simplex_rows,
    renyi_divergence,
    safe_divide,
    safe_exp,
    safe_log,
    sparse_block_diag,
    sparse_max_abs,
    sparse_nnz_fraction,
    sparse_row_norms,
    sparse_vstack_incremental,
    spectral_norm,
    symmetrise,
    tolerance_margin,
    total_variation,
    weighted_mae,
    weighted_mse,
)


# =========================================================================
# 1. Sparse matrix operations
# =========================================================================


class TestBuildCsr:
    """Tests for build_csr()."""

    def test_basic_construction(self):
        A = build_csr([1.0, -1.0], [0, 0], [0, 1], (1, 3))
        expected = np.array([[1.0, -1.0, 0.0]])
        np.testing.assert_array_equal(A.toarray(), expected)

    def test_empty_matrix(self):
        A = build_csr([], [], [], (3, 3))
        assert A.shape == (3, 3)
        assert A.nnz == 0

    def test_identity_matrix(self):
        n = 4
        data = [1.0] * n
        rows = list(range(n))
        cols = list(range(n))
        A = build_csr(data, rows, cols, (n, n))
        np.testing.assert_array_equal(A.toarray(), np.eye(n))

    def test_duplicate_entries_summed(self):
        A = build_csr([1.0, 2.0], [0, 0], [0, 0], (1, 1))
        assert A[0, 0] == 3.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            build_csr([1.0], [0, 0], [0], (1, 1))

    def test_row_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="row_ind"):
            build_csr([1.0], [5], [0], (3, 3))

    def test_col_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="col_ind"):
            build_csr([1.0], [0], [5], (3, 3))

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            build_csr([1.0], [-1], [0], (3, 3))

    def test_result_is_csr(self):
        A = build_csr([1.0], [0], [0], (2, 2))
        assert isinstance(A, sparse.csr_matrix)

    def test_sorted_indices(self):
        A = build_csr([2.0, 1.0], [0, 0], [1, 0], (1, 3))
        assert A.has_sorted_indices


class TestBuildCsrFromDense:
    """Tests for build_csr_from_dense()."""

    def test_roundtrip(self):
        M = np.array([[1.0, 0.0], [0.0, 2.0]])
        S = build_csr_from_dense(M)
        np.testing.assert_array_equal(S.toarray(), M)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            build_csr_from_dense(np.array([1, 2, 3]))

    def test_preserves_zeros(self):
        M = np.zeros((3, 3))
        S = build_csr_from_dense(M)
        assert S.nnz == 0

    def test_dtype_conversion(self):
        M = np.array([[1, 2], [3, 4]], dtype=np.int32)
        S = build_csr_from_dense(M)
        assert S.dtype == np.float64


class TestSparseVstackIncremental:
    """Tests for sparse_vstack_incremental()."""

    def test_none_base_returns_new_rows(self):
        rows = sparse.csr_matrix(np.array([[1.0, 2.0]]))
        result = sparse_vstack_incremental(None, rows)
        np.testing.assert_array_equal(result.toarray(), rows.toarray())

    def test_stacking_two_blocks(self):
        base = sparse.csr_matrix(np.array([[1.0, 0.0]]))
        new = sparse.csr_matrix(np.array([[0.0, 2.0]]))
        result = sparse_vstack_incremental(base, new)
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(result.toarray(), expected)

    def test_column_mismatch_raises(self):
        base = sparse.csr_matrix(np.array([[1.0, 0.0]]))
        new = sparse.csr_matrix(np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="Column mismatch"):
            sparse_vstack_incremental(base, new)

    def test_incremental_growth(self):
        result = None
        for i in range(5):
            row = sparse.csr_matrix(np.array([[float(i), float(i + 1)]]))
            result = sparse_vstack_incremental(result, row)
        assert result.shape == (5, 2)


class TestSparseBlockDiag:
    """Tests for sparse_block_diag()."""

    def test_single_block(self):
        block = np.eye(3)
        result = sparse_block_diag([block])
        np.testing.assert_array_equal(result.toarray(), np.eye(3))

    def test_two_blocks(self):
        b1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        b2 = np.array([[5.0]])
        result = sparse_block_diag([b1, b2])
        expected = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ])
        np.testing.assert_array_equal(result.toarray(), expected)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            sparse_block_diag([])

    def test_non_2d_block_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            sparse_block_diag([np.array([1, 2, 3])])

    def test_mixed_dense_sparse(self):
        dense = np.eye(2)
        sp = sparse.csr_matrix(np.array([[3.0, 4.0], [5.0, 6.0]]))
        result = sparse_block_diag([dense, sp])
        assert result.shape == (4, 4)
        assert result[0, 0] == 1.0
        assert result[2, 2] == 3.0


class TestSparseNnzFraction:
    """Tests for sparse_nnz_fraction()."""

    def test_identity(self):
        A = sparse.eye(10, format="csr")
        frac = sparse_nnz_fraction(A)
        assert abs(frac - 0.1) < 1e-12

    def test_full_matrix(self):
        A = sparse.csr_matrix(np.ones((3, 3)))
        assert abs(sparse_nnz_fraction(A) - 1.0) < 1e-12

    def test_empty_matrix(self):
        A = sparse.csr_matrix((0, 5))
        assert sparse_nnz_fraction(A) == 0.0


class TestSparseRowNorms:
    """Tests for sparse_row_norms()."""

    def test_identity_l2(self):
        A = sparse.eye(3, format="csr")
        norms = sparse_row_norms(A, ord=2)
        np.testing.assert_allclose(norms, np.ones(3))

    def test_l1_norm(self):
        A = sparse.csr_matrix(np.array([[1.0, -2.0], [3.0, 0.0]]))
        norms = sparse_row_norms(A, ord=1)
        np.testing.assert_allclose(norms, [3.0, 3.0])

    def test_linf_norm(self):
        A = sparse.csr_matrix(np.array([[1.0, -5.0], [3.0, 0.0]]))
        norms = sparse_row_norms(A, ord=np.inf)
        np.testing.assert_allclose(norms, [5.0, 3.0])


class TestSparseMaxAbs:
    """Tests for sparse_max_abs()."""

    def test_basic(self):
        A = sparse.csr_matrix(np.array([[1.0, -3.0], [2.0, 0.0]]))
        assert sparse_max_abs(A) == 3.0

    def test_empty_matrix(self):
        A = sparse.csr_matrix((3, 3))
        assert sparse_max_abs(A) == 0.0


# =========================================================================
# 2. Numerical stability primitives
# =========================================================================


class TestLogSumExp:
    """Tests for log_sum_exp()."""

    def test_singleton(self):
        """log_sum_exp([x]) should equal x."""
        assert abs(log_sum_exp(np.array([5.0])) - 5.0) < 1e-12

    @pytest.mark.parametrize("x", [0.0, -10.0, 42.0])
    def test_singleton_parametrized(self, x):
        assert abs(log_sum_exp(np.array([x])) - x) < 1e-12

    def test_two_equal_values(self):
        """log_sum_exp([x, x]) = x + log(2)."""
        x = 3.0
        expected = x + np.log(2.0)
        assert abs(log_sum_exp(np.array([x, x])) - expected) < 1e-12

    def test_large_values_no_overflow(self):
        """Should handle very large values without overflow."""
        result = log_sum_exp(np.array([1000.0, 1001.0]))
        expected = 1001.0 + np.log(1.0 + np.exp(-1.0))
        assert abs(result - expected) < 1e-8

    def test_very_negative_values(self):
        result = log_sum_exp(np.array([-1000.0, -999.0]))
        expected = -999.0 + np.log(1.0 + np.exp(-1.0))
        assert abs(result - expected) < 1e-8

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            log_sum_exp(np.array([]))

    def test_inf_value(self):
        result = log_sum_exp(np.array([np.inf, 1.0]))
        assert result == np.inf

    def test_neg_inf_value(self):
        result = log_sum_exp(np.array([-np.inf, 0.0]))
        assert abs(result - 0.0) < 1e-12

    def test_uniform_array(self):
        """log_sum_exp of n equal values = x + log(n)."""
        n = 100
        x = 2.5
        expected = x + np.log(n)
        result = log_sum_exp(np.full(n, x))
        assert abs(result - expected) < 1e-10


class TestLogSumExpArray:
    """Tests for log_sum_exp_array()."""

    def test_2d_axis0(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = log_sum_exp_array(a, axis=0)
        assert result.shape == (2,)
        expected_0 = log_sum_exp(np.array([1.0, 3.0]))
        assert abs(result[0] - expected_0) < 1e-12

    def test_2d_axis1(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = log_sum_exp_array(a, axis=1)
        assert result.shape == (2,)
        expected_0 = log_sum_exp(np.array([1.0, 2.0]))
        assert abs(result[0] - expected_0) < 1e-12

    def test_large_values(self):
        a = np.array([[1000.0, 1001.0], [2000.0, 2001.0]])
        result = log_sum_exp_array(a, axis=1)
        assert np.all(np.isfinite(result))


class TestLogSubtractExp:
    """Tests for log_subtract_exp()."""

    def test_basic(self):
        result = log_subtract_exp(2.0, 1.0)
        expected = np.log(np.exp(2.0) - np.exp(1.0))
        assert abs(result - expected) < 1e-10

    def test_equal_returns_neg_inf(self):
        result = log_subtract_exp(5.0, 5.0)
        assert result == -np.inf

    def test_b_greater_raises(self):
        with pytest.raises(ValueError, match="a >= b"):
            log_subtract_exp(1.0, 2.0)

    def test_b_neg_inf(self):
        """log(exp(a) - exp(-inf)) = log(exp(a)) = a."""
        result = log_subtract_exp(3.0, -np.inf)
        assert abs(result - 3.0) < 1e-12

    def test_large_values_stable(self):
        result = log_subtract_exp(1000.0, 999.0)
        expected = 999.0 + np.log(np.exp(1.0) - 1.0)
        assert abs(result - expected) < 1e-8


class TestSafeDivide:
    """Tests for safe_divide()."""

    def test_normal_division(self):
        result = safe_divide(6.0, 3.0)
        assert abs(result - 2.0) < 1e-12

    def test_zero_denominator_returns_fill(self):
        result = safe_divide(5.0, 0.0)
        assert result == 0.0

    def test_custom_fill(self):
        result = safe_divide(5.0, 0.0, fill=-1.0)
        assert result == -1.0

    def test_array_division(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 0.0, 3.0])
        result = safe_divide(a, b)
        np.testing.assert_allclose(result, [1.0, 0.0, 1.0])

    def test_near_zero_denominator(self):
        result = safe_divide(1.0, 1e-310)
        assert result == 0.0

    def test_negative_numerator_zero_denominator(self):
        result = safe_divide(-5.0, 0.0, fill=99.0)
        assert result == 99.0

    def test_both_arrays(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = safe_divide(a, b)
        np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 2.0]])


class TestSafeLog:
    """Tests for safe_log()."""

    def test_positive_values(self):
        result = safe_log(np.e)
        assert abs(result - 1.0) < 1e-12

    def test_zero_uses_floor(self):
        result = safe_log(0.0)
        assert np.isfinite(result)
        assert result < -600  # log(1e-300) ≈ -690

    def test_negative_uses_floor(self):
        result = safe_log(-1.0)
        assert np.isfinite(result)

    def test_array_input(self):
        result = safe_log(np.array([1.0, 0.0, np.e]))
        assert np.all(np.isfinite(result))
        assert abs(result[0] - 0.0) < 1e-12
        assert abs(result[2] - 1.0) < 1e-12

    def test_custom_floor(self):
        result = safe_log(0.0, floor=1e-10)
        expected = np.log(1e-10)
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 100.0])
    def test_matches_log_for_positive(self, x):
        assert abs(safe_log(x) - np.log(x)) < 1e-12


class TestSafeExp:
    """Tests for safe_exp()."""

    def test_normal_value(self):
        result = safe_exp(1.0)
        assert abs(result - np.e) < 1e-12

    def test_large_value_capped(self):
        result = safe_exp(1000.0)
        expected = np.exp(700.0)
        assert abs(result - expected) < 1e-6 * expected

    def test_negative_value(self):
        result = safe_exp(-5.0)
        assert abs(result - np.exp(-5.0)) < 1e-12

    def test_array_input(self):
        result = safe_exp(np.array([0.0, 800.0, -1.0]))
        assert np.all(np.isfinite(result))
        assert abs(result[0] - 1.0) < 1e-12

    def test_custom_cap(self):
        result = safe_exp(100.0, cap=50.0)
        assert abs(result - np.exp(50.0)) < 1e-6 * np.exp(50.0)


# =========================================================================
# 3. Divergence measures
# =========================================================================


class TestKLDivergence:
    """Tests for kl_divergence()."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(kl_divergence(p, p)) < 1e-12

    def test_known_value(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        assert abs(kl_divergence(p, q) - expected) < 1e-10

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        assert kl_divergence(p, q) >= -1e-12

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            kl_divergence(np.array([0.5, 0.5]), np.array([1.0]))

    def test_with_zeros(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        result = kl_divergence(p, q)
        assert np.isfinite(result)


class TestTotalVariation:
    """Tests for total_variation()."""

    def test_identical(self):
        p = np.array([0.5, 0.5])
        assert abs(total_variation(p, p)) < 1e-12

    def test_disjoint(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert abs(total_variation(p, q) - 1.0) < 1e-12

    def test_known_value(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        expected = 0.5 * (abs(0.3 - 0.5) + abs(0.7 - 0.5))
        assert abs(total_variation(p, q) - expected) < 1e-12

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            total_variation(np.array([0.5, 0.5]), np.array([1.0]))

    def test_bounded_by_one(self):
        rng = np.random.default_rng(99)
        p = rng.dirichlet(np.ones(10))
        q = rng.dirichlet(np.ones(10))
        tv = total_variation(p, q)
        assert 0.0 <= tv <= 1.0 + 1e-12


class TestRenyiDivergence:
    """Tests for renyi_divergence()."""

    def test_alpha_near_one_equals_kl(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        kl = kl_divergence(p, q)
        renyi = renyi_divergence(p, q, alpha=1.0 + 1e-14)
        assert abs(renyi - kl) < 1e-6

    def test_alpha_two(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        result = renyi_divergence(p, q, alpha=2.0)
        assert np.isfinite(result)
        assert result >= -1e-12

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(renyi_divergence(p, p, alpha=2.0)) < 1e-10

    def test_invalid_alpha_raises(self):
        p = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="alpha must be > 0"):
            renyi_divergence(p, p, alpha=-1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            renyi_divergence(np.array([0.5, 0.5]), np.array([1.0]), alpha=2.0)


class TestEntropy:
    """Tests for entropy()."""

    def test_uniform_distribution(self):
        n = 8
        p = np.ones(n) / n
        expected = np.log(n)
        assert abs(entropy(p) - expected) < 1e-10

    def test_deterministic(self):
        p = np.array([1.0, 0.0, 0.0])
        result = entropy(p)
        assert abs(result) < 1e-6

    def test_binary_entropy(self):
        p = np.array([0.5, 0.5])
        assert abs(entropy(p) - np.log(2)) < 1e-12

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        p = rng.dirichlet(np.ones(20))
        assert entropy(p) >= -1e-12

    def test_with_zeros_finite(self):
        p = np.array([0.5, 0.0, 0.5])
        assert np.isfinite(entropy(p))


# =========================================================================
# 4. Tolerance management
# =========================================================================


class TestComputeDpTolerance:
    """Tests for compute_dp_tolerance()."""

    def test_basic(self):
        result = compute_dp_tolerance(epsilon=1.0, solver_tol=1e-8)
        expected = math.exp(1.0) * 1e-8
        assert abs(result - expected) < 1e-20

    def test_large_epsilon(self):
        result = compute_dp_tolerance(epsilon=5.0, solver_tol=1e-6)
        expected = math.exp(5.0) * 1e-6
        assert abs(result - expected) < 1e-15

    def test_zero_epsilon_raises(self):
        with pytest.raises(ConfigurationError):
            compute_dp_tolerance(epsilon=0.0, solver_tol=1e-8)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ConfigurationError):
            compute_dp_tolerance(epsilon=-1.0, solver_tol=1e-8)

    def test_zero_solver_tol_raises(self):
        with pytest.raises(ConfigurationError):
            compute_dp_tolerance(epsilon=1.0, solver_tol=0.0)

    def test_negative_solver_tol_raises(self):
        with pytest.raises(ConfigurationError):
            compute_dp_tolerance(epsilon=1.0, solver_tol=-1e-8)


class TestCheckToleranceConsistency:
    """Tests for check_tolerance_consistency()."""

    def test_consistent(self):
        dp_tol = math.exp(1.0) * 1e-8
        assert check_tolerance_consistency(dp_tol, 1e-8, 1.0) is True

    def test_inconsistent(self):
        dp_tol = 1e-10
        assert check_tolerance_consistency(dp_tol, 1e-8, 1.0) is False

    def test_exact_boundary(self):
        dp_tol = math.exp(1.0) * 1e-8
        assert check_tolerance_consistency(dp_tol, 1e-8, 1.0) is True

    def test_with_slack(self):
        dp_tol = 10 * math.exp(1.0) * 1e-8
        assert check_tolerance_consistency(dp_tol, 1e-8, 1.0) is True


class TestAdaptiveTolerance:
    """Tests for adaptive_tolerance()."""

    def test_iteration_zero(self):
        result = adaptive_tolerance(0, base_tol=1e-6)
        assert abs(result - 1e-6) < 1e-18

    def test_decay(self):
        base = 1e-6
        rate = 0.9
        result = adaptive_tolerance(1, base_tol=base, decay_rate=rate)
        assert abs(result - base * rate) < 1e-18

    def test_floor(self):
        result = adaptive_tolerance(
            1000, base_tol=1e-6, decay_rate=0.5, min_tol=1e-12
        )
        assert abs(result - 1e-12) < 1e-24

    def test_negative_iteration_raises(self):
        with pytest.raises(ValueError, match="iteration"):
            adaptive_tolerance(-1, base_tol=1e-6)

    def test_zero_base_tol_raises(self):
        with pytest.raises(ValueError, match="base_tol"):
            adaptive_tolerance(0, base_tol=0.0)

    def test_invalid_decay_rate_raises(self):
        with pytest.raises(ValueError, match="decay_rate"):
            adaptive_tolerance(0, base_tol=1e-6, decay_rate=1.5)

    def test_zero_decay_rate_raises(self):
        with pytest.raises(ValueError, match="decay_rate"):
            adaptive_tolerance(0, base_tol=1e-6, decay_rate=0.0)

    def test_zero_min_tol_raises(self):
        with pytest.raises(ValueError, match="min_tol"):
            adaptive_tolerance(0, base_tol=1e-6, min_tol=0.0)

    @pytest.mark.parametrize("iteration", [0, 1, 5, 10, 50])
    def test_monotonically_decreasing(self, iteration):
        t0 = adaptive_tolerance(iteration, base_tol=1.0, decay_rate=0.9)
        t1 = adaptive_tolerance(iteration + 1, base_tol=1.0, decay_rate=0.9)
        assert t1 <= t0 + 1e-15


class TestToleranceMargin:
    """Tests for tolerance_margin()."""

    def test_positive_margin(self):
        required = math.exp(1.0) * 1e-8
        margin = tolerance_margin(required * 2, 1e-8, 1.0)
        assert margin > 0

    def test_zero_margin(self):
        required = math.exp(1.0) * 1e-8
        margin = tolerance_margin(required, 1e-8, 1.0)
        assert abs(margin) < 1e-18

    def test_negative_margin(self):
        margin = tolerance_margin(1e-12, 1e-8, 1.0)
        assert margin < 0


# =========================================================================
# 5. Condition number analysis
# =========================================================================


class TestEstimateCondition:
    """Tests for estimate_condition()."""

    def test_identity_dense(self):
        I = np.eye(5)
        cond = estimate_condition(I)
        assert abs(cond - 1.0) < 1e-8

    def test_identity_sparse(self):
        I = sparse.eye(5, format="csr")
        cond = estimate_condition(I)
        assert abs(cond - 1.0) < 1e-6

    def test_ill_conditioned(self):
        A = np.diag([1.0, 1e-10])
        cond = estimate_condition(A)
        assert cond > 1e9

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_condition(np.array([]).reshape(0, 0))

    def test_norm_method_sparse(self):
        A = sparse.eye(4, format="csr")
        cond = estimate_condition(A, method="norm")
        assert abs(cond - 1.0) < 1e-12

    def test_condition_ge_one(self):
        rng = np.random.default_rng(12)
        A = rng.standard_normal((5, 5))
        cond = estimate_condition(A)
        assert cond >= 1.0 - 1e-8


class TestDiagonalPreconditioning:
    """Tests for diagonal_preconditioning()."""

    def test_dense_unit_rows(self):
        A = np.array([[3.0, 4.0], [0.0, 5.0]])
        A_scaled, norms = diagonal_preconditioning(A)
        row_norms = np.linalg.norm(A_scaled, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(2), atol=1e-12)

    def test_sparse_input(self):
        A = sparse.csr_matrix(np.array([[3.0, 4.0], [0.0, 5.0]]))
        A_scaled, norms = diagonal_preconditioning(A)
        row_norms_result = sparse_row_norms(A_scaled, ord=2)
        np.testing.assert_allclose(row_norms_result, np.ones(2), atol=1e-12)

    def test_zero_row_unchanged(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        A_scaled, norms = diagonal_preconditioning(A)
        np.testing.assert_allclose(A_scaled[1], [0.0, 0.0], atol=1e-15)


class TestDetectNearSingularity:
    """Tests for detect_near_singularity()."""

    def test_identity_not_singular(self):
        assert detect_near_singularity(np.eye(5)) is False

    def test_near_singular(self):
        A = np.diag([1.0, 1e-16])
        assert detect_near_singularity(A) is True

    def test_custom_threshold(self):
        A = np.diag([1.0, 0.01])
        assert detect_near_singularity(A, threshold=10.0) is True
        assert detect_near_singularity(A, threshold=1000.0) is False


class TestCheckConditionAndRaise:
    """Tests for check_condition_and_raise()."""

    def test_good_matrix(self):
        A = np.eye(5)
        cond = check_condition_and_raise(A, max_condition=100.0)
        assert cond < 100.0

    def test_bad_matrix_raises(self):
        A = np.diag([1.0, 1e-16])
        with pytest.raises(NumericalInstabilityError):
            check_condition_and_raise(A, max_condition=1e10)


# =========================================================================
# 6. Matrix utilities and projections
# =========================================================================


class TestIsStochastic:
    """Tests for is_stochastic()."""

    def test_uniform(self):
        M = np.ones((3, 3)) / 3
        assert is_stochastic(M) is True

    def test_identity(self):
        assert is_stochastic(np.eye(3)) is True

    def test_negative_entries(self):
        M = np.array([[0.5, -0.1, 0.6], [0.3, 0.3, 0.4]])
        assert is_stochastic(M) is False

    def test_wrong_row_sum(self):
        M = np.array([[0.5, 0.4], [0.3, 0.3]])
        assert is_stochastic(M) is False

    def test_1d_fails(self):
        assert is_stochastic(np.array([0.5, 0.5])) is False

    def test_rectangular(self):
        M = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        assert is_stochastic(M) is True


class TestIsDoublyStochastic:
    """Tests for is_doubly_stochastic()."""

    def test_permutation(self):
        M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert is_doubly_stochastic(M) is True

    def test_uniform(self):
        n = 4
        M = np.ones((n, n)) / n
        assert is_doubly_stochastic(M) is True

    def test_identity(self):
        assert is_doubly_stochastic(np.eye(3)) is True

    def test_rectangular_fails(self):
        M = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])
        assert is_doubly_stochastic(M) is False

    def test_row_but_not_col_stochastic(self):
        M = np.array([[0.7, 0.3], [0.7, 0.3]])
        assert is_doubly_stochastic(M) is False


class TestProjectSimplex:
    """Tests for project_simplex()."""

    def test_already_on_simplex(self):
        p = np.array([0.2, 0.3, 0.5])
        result = project_simplex(p)
        np.testing.assert_allclose(result, p, atol=1e-12)

    def test_uniform_excess(self):
        v = np.array([0.5, 0.5, 0.5])
        result = project_simplex(v)
        assert abs(np.sum(result) - 1.0) < 1e-12
        assert np.all(result >= -1e-15)

    def test_negative_entries(self):
        v = np.array([-1.0, 2.0, 0.5])
        result = project_simplex(v)
        assert abs(np.sum(result) - 1.0) < 1e-12
        assert np.all(result >= -1e-15)

    def test_single_element(self):
        result = project_simplex(np.array([5.0]))
        assert abs(result[0] - 1.0) < 1e-12

    def test_empty(self):
        result = project_simplex(np.array([]))
        assert len(result) == 0

    def test_all_negative(self):
        v = np.array([-5.0, -3.0, -1.0])
        result = project_simplex(v)
        assert abs(np.sum(result) - 1.0) < 1e-12
        assert np.all(result >= -1e-15)

    def test_large_vector(self):
        rng = np.random.default_rng(123)
        v = rng.standard_normal(1000)
        result = project_simplex(v)
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert np.all(result >= -1e-15)

    @pytest.mark.parametrize("n", [2, 5, 10, 50])
    def test_projection_is_closest_point(self, n):
        """Projected point is closer to simplex than original."""
        rng = np.random.default_rng(n)
        v = rng.standard_normal(n) * 3
        p = project_simplex(v)
        assert abs(np.sum(p) - 1.0) < 1e-10
        assert np.all(p >= -1e-15)

    def test_idempotent(self):
        """Projecting a simplex point again gives the same point."""
        v = np.array([0.1, 0.2, 0.3, 0.4])
        p1 = project_simplex(v)
        p2 = project_simplex(p1)
        np.testing.assert_allclose(p1, p2, atol=1e-12)


class TestProjectSimplexRows:
    """Tests for project_simplex_rows()."""

    def test_basic(self):
        M = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
        result = project_simplex_rows(M)
        for row in result:
            assert abs(np.sum(row) - 1.0) < 1e-12
            assert np.all(row >= -1e-15)

    def test_already_stochastic(self):
        M = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
        result = project_simplex_rows(M)
        np.testing.assert_allclose(result, M, atol=1e-12)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            project_simplex_rows(np.array([1, 2, 3]))


class TestProjectPsd:
    """Tests for project_psd()."""

    def test_already_psd(self):
        M = np.eye(3)
        result = project_psd(M)
        np.testing.assert_allclose(result, M, atol=1e-12)

    def test_negative_eigenvalue_clipped(self):
        eigenvalues = np.array([2.0, -1.0, 0.5])
        Q = np.eye(3)  # already diagonal
        M = np.diag(eigenvalues)
        result = project_psd(M)
        result_eigs = np.linalg.eigvalsh(result)
        assert np.all(result_eigs >= -1e-12)

    def test_result_is_symmetric(self):
        rng = np.random.default_rng(42)
        M = rng.standard_normal((4, 4))
        result = project_psd(M)
        assert np.allclose(result, result.T, atol=1e-12)

    def test_result_is_psd(self):
        rng = np.random.default_rng(7)
        M = rng.standard_normal((5, 5))
        result = project_psd(M)
        eigs = np.linalg.eigvalsh(result)
        assert np.all(eigs >= -1e-10)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            project_psd(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_zero_matrix(self):
        M = np.zeros((3, 3))
        result = project_psd(M)
        np.testing.assert_allclose(result, np.zeros((3, 3)), atol=1e-15)

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_psd_projection_random(self, n):
        rng = np.random.default_rng(n + 100)
        M = rng.standard_normal((n, n))
        M = M + M.T  # symmetric but not necessarily PSD
        result = project_psd(M)
        eigs = np.linalg.eigvalsh(result)
        assert np.all(eigs >= -1e-10)
        assert abs(np.sum(result - result.T)) < 1e-12


class TestIsPsd:
    """Tests for is_psd()."""

    def test_identity(self):
        assert is_psd(np.eye(3)) is True

    def test_negative_definite(self):
        assert is_psd(-np.eye(3)) is False

    def test_indefinite(self):
        M = np.diag([1.0, -0.5])
        assert is_psd(M) is False

    def test_positive_semidefinite(self):
        M = np.diag([1.0, 0.0])
        assert is_psd(M) is True

    def test_non_square_fails(self):
        assert is_psd(np.array([[1, 2, 3]])) is False

    def test_1d_fails(self):
        assert is_psd(np.array([1.0, 2.0])) is False

    def test_small_negative_eigenvalue_within_tol(self):
        M = np.diag([1.0, -1e-11])
        assert is_psd(M, tol=-1e-10) is True


class TestIsSymmetric:
    """Tests for is_symmetric()."""

    def test_symmetric_matrix(self):
        M = np.array([[1, 2], [2, 3]], dtype=float)
        assert is_symmetric(M) is True

    def test_asymmetric_matrix(self):
        M = np.array([[1, 2], [3, 4]], dtype=float)
        assert is_symmetric(M) is False

    def test_non_square(self):
        assert is_symmetric(np.array([[1, 2, 3]])) is False

    def test_identity(self):
        assert is_symmetric(np.eye(5)) is True


class TestNormalizeRows:
    """Tests for normalize_rows()."""

    def test_basic(self):
        M = np.array([[2.0, 2.0], [1.0, 3.0]])
        result = normalize_rows(M)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-12)

    def test_with_floor(self):
        M = np.array([[1.0, -1.0], [2.0, 2.0]])
        result = normalize_rows(M, floor=0.0)
        assert np.all(result >= 0.0)

    def test_zero_row(self):
        M = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = normalize_rows(M)
        # Zero row stays zero (divided by safe_sum = 1)
        np.testing.assert_allclose(result[0], [0.0, 0.0])
        np.testing.assert_allclose(result[1], [0.5, 0.5], atol=1e-12)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            normalize_rows(np.array([1, 2, 3]))


class TestSymmetrise:
    """Tests for symmetrise()."""

    def test_already_symmetric(self):
        M = np.array([[1, 2], [2, 3]], dtype=float)
        np.testing.assert_array_equal(symmetrise(M), M)

    def test_asymmetric(self):
        M = np.array([[1, 4], [0, 3]], dtype=float)
        result = symmetrise(M)
        expected = np.array([[1, 2], [2, 3]], dtype=float)
        np.testing.assert_allclose(result, expected)


# =========================================================================
# 7. Error metrics
# =========================================================================


class TestWeightedMSE:
    """Tests for weighted_mse()."""

    def test_perfect_mechanism(self):
        mechanism = np.eye(3)
        query_values = np.array([0.0, 1.0, 2.0])
        y_grid = np.array([0.0, 1.0, 2.0])
        mse = weighted_mse(mechanism, query_values, y_grid)
        assert abs(mse) < 1e-12

    def test_nonzero_error(self):
        mechanism = np.array([[0.5, 0.5], [0.5, 0.5]])
        query_values = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        mse = weighted_mse(mechanism, query_values, y_grid)
        assert mse > 0

    def test_custom_weights(self):
        mechanism = np.array([[1.0, 0.0], [0.0, 1.0]])
        query_values = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        mse = weighted_mse(mechanism, query_values, y_grid, weights=weights)
        assert abs(mse) < 1e-12

    def test_all_weight_on_one(self):
        mechanism = np.array([[0.0, 1.0], [1.0, 0.0]])
        query_values = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        weights = np.array([1.0, 0.0])
        mse = weighted_mse(mechanism, query_values, y_grid, weights=weights)
        # Input 0 maps to y=1.0, error = (0 - 1)^2 = 1.0
        assert abs(mse - 1.0) < 1e-12


class TestWeightedMAE:
    """Tests for weighted_mae()."""

    def test_perfect_mechanism(self):
        mechanism = np.eye(3)
        query_values = np.array([0.0, 1.0, 2.0])
        y_grid = np.array([0.0, 1.0, 2.0])
        mae = weighted_mae(mechanism, query_values, y_grid)
        assert abs(mae) < 1e-12

    def test_nonzero_error(self):
        mechanism = np.array([[0.5, 0.5], [0.5, 0.5]])
        query_values = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        mae = weighted_mae(mechanism, query_values, y_grid)
        assert mae > 0

    def test_mae_le_mse_sqrt(self):
        """MAE is generally <= sqrt(MSE) for uniform distributions."""
        rng = np.random.default_rng(88)
        n, k = 5, 5
        mechanism = rng.dirichlet(np.ones(k), size=n)
        query_values = np.arange(n, dtype=float)
        y_grid = np.linspace(-1, n, k)
        mae = weighted_mae(mechanism, query_values, y_grid)
        assert mae >= 0


class TestLinfError:
    """Tests for linf_error()."""

    def test_perfect_mechanism(self):
        mechanism = np.eye(3)
        query_values = np.array([0.0, 1.0, 2.0])
        y_grid = np.array([0.0, 1.0, 2.0])
        err = linf_error(mechanism, query_values, y_grid)
        assert abs(err) < 1e-12

    def test_worst_case(self):
        mechanism = np.array([[0.0, 1.0], [1.0, 0.0]])
        query_values = np.array([0.0, 10.0])
        y_grid = np.array([0.0, 10.0])
        err = linf_error(mechanism, query_values, y_grid)
        assert abs(err - 10.0) < 1e-12

    def test_non_negative(self):
        rng = np.random.default_rng(55)
        n, k = 4, 4
        mechanism = rng.dirichlet(np.ones(k), size=n)
        query_values = np.arange(n, dtype=float)
        y_grid = np.linspace(0, n - 1, k)
        assert linf_error(mechanism, query_values, y_grid) >= 0


# =========================================================================
# 8. Output grid and privacy loss
# =========================================================================


class TestComputeOutputGrid:
    """Tests for compute_output_grid()."""

    def test_basic(self):
        query_values = np.array([0.0, 1.0, 2.0])
        grid = compute_output_grid(query_values, k=5)
        assert len(grid) == 5
        assert grid[0] < 0.0  # padded below
        assert grid[-1] > 2.0  # padded above

    def test_single_value(self):
        grid = compute_output_grid(np.array([5.0]), k=10)
        assert len(grid) == 10
        assert grid[0] < 5.0
        assert grid[-1] > 5.0

    def test_custom_padding(self):
        grid = compute_output_grid(
            np.array([0.0, 1.0]), k=3, padding_factor=0.0
        )
        np.testing.assert_allclose(grid, [0.0, 0.5, 1.0])

    def test_custom_sensitivity(self):
        grid = compute_output_grid(
            np.array([0.0, 1.0]), k=3, sensitivity=2.0
        )
        assert grid[0] < -5.0  # 0 - 3*2 = -6
        assert grid[-1] > 6.0   # 1 + 3*2 = 7


class TestPrivacyLossRV:
    """Tests for privacy_loss_rv()."""

    def test_identical_rows(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        losses, probs = privacy_loss_rv(p, p)
        np.testing.assert_allclose(losses, np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(probs, p, atol=1e-12)

    def test_output_shapes(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        losses, probs = privacy_loss_rv(p, q)
        assert losses.shape == (2,)
        assert probs.shape == (2,)

    def test_finite_with_zeros(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])
        losses, probs = privacy_loss_rv(p, q)
        assert np.all(np.isfinite(losses))


# =========================================================================
# 9. Additional matrix analysis utilities
# =========================================================================


class TestFrobeniusNorm:
    """Tests for frobenius_norm()."""

    def test_identity(self):
        assert abs(frobenius_norm(np.eye(3)) - np.sqrt(3)) < 1e-12

    def test_zero_matrix(self):
        assert abs(frobenius_norm(np.zeros((3, 3)))) < 1e-15

    def test_known_value(self):
        M = np.array([[1, 2], [3, 4]], dtype=float)
        expected = np.sqrt(1 + 4 + 9 + 16)
        assert abs(frobenius_norm(M) - expected) < 1e-12


class TestSpectralNorm:
    """Tests for spectral_norm()."""

    def test_identity(self):
        assert abs(spectral_norm(np.eye(3)) - 1.0) < 1e-12

    def test_empty(self):
        assert spectral_norm(np.array([]).reshape(0, 0)) == 0.0

    def test_diagonal(self):
        M = np.diag([3.0, 1.0, 2.0])
        assert abs(spectral_norm(M) - 3.0) < 1e-12


class TestMatrixRankEstimate:
    """Tests for matrix_rank_estimate()."""

    def test_full_rank(self):
        assert matrix_rank_estimate(np.eye(5)) == 5

    def test_rank_deficient(self):
        M = np.array([[1, 2], [2, 4]], dtype=float)
        assert matrix_rank_estimate(M) == 1

    def test_zero_matrix(self):
        assert matrix_rank_estimate(np.zeros((3, 3))) == 0

    def test_sparse_input(self):
        A = sparse.eye(4, format="csr")
        assert matrix_rank_estimate(A) == 4


# =========================================================================
# 10. Integration / cross-function tests
# =========================================================================


class TestIntegration:
    """Cross-function integration tests."""

    def test_simplex_projection_is_stochastic(self):
        """After projecting rows onto simplex, matrix should be stochastic."""
        rng = np.random.default_rng(42)
        M = rng.standard_normal((5, 4))
        M_proj = project_simplex_rows(M)
        assert is_stochastic(M_proj)

    def test_psd_projection_passes_is_psd(self):
        """project_psd output should pass is_psd check."""
        rng = np.random.default_rng(99)
        M = rng.standard_normal((4, 4))
        M_psd = project_psd(M)
        assert is_psd(M_psd)

    def test_normalize_rows_is_stochastic(self):
        """normalize_rows with non-negative input produces stochastic matrix."""
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_rows(M)
        assert is_stochastic(result)

    def test_kl_and_total_variation_consistency(self):
        """Pinsker: TV <= sqrt(KL/2)."""
        rng = np.random.default_rng(7)
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        tv = total_variation(p, q)
        kl = kl_divergence(p, q)
        assert tv <= np.sqrt(kl / 2) + 1e-10

    def test_tolerance_roundtrip(self):
        """compute_dp_tolerance output passes check_tolerance_consistency."""
        eps = 2.0
        solver_tol = 1e-8
        dp_tol = compute_dp_tolerance(eps, solver_tol)
        assert check_tolerance_consistency(dp_tol, solver_tol, eps)

    def test_sparse_build_condition_estimate(self):
        """Build sparse identity and verify condition ≈ 1."""
        n = 10
        data = [1.0] * n
        rows = list(range(n))
        cols = list(range(n))
        A = build_csr(data, rows, cols, (n, n))
        cond = estimate_condition(A, method="norm")
        assert abs(cond - 1.0) < 1e-12

    def test_entropy_of_normalized_rows(self):
        """Entropy is well-defined on normalized rows."""
        M = np.array([[1.0, 3.0], [2.0, 2.0]])
        normed = normalize_rows(M)
        for row in normed:
            h = entropy(row)
            assert h >= 0
            assert h <= np.log(len(row)) + 1e-10

    def test_safe_log_exp_roundtrip(self):
        """safe_exp(safe_log(x)) ≈ x for positive x."""
        x = np.array([0.1, 1.0, 10.0, 100.0])
        roundtrip = safe_exp(safe_log(x))
        np.testing.assert_allclose(roundtrip, x, rtol=1e-10)

    def test_block_diag_vstack(self):
        """Block diag then vstack should work without errors."""
        b1 = np.eye(2)
        b2 = np.eye(2)
        bd = sparse_block_diag([b1, b2])
        extra = sparse.csr_matrix(np.ones((1, 4)))
        result = sparse_vstack_incremental(bd, extra)
        assert result.shape == (5, 4)

    def test_doubly_stochastic_permutation_entropy(self):
        """Permutation matrix: doubly stochastic, each row has zero entropy."""
        M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert is_doubly_stochastic(M)
        for row in M:
            h = entropy(row)
            assert h < 1e-6
