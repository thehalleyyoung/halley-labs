"""
Comprehensive tests for dp_forge.lp_builder — the LP construction engine.

Covers: VariableLayout, output grid, loss matrix, pure/approx DP constraints,
simplex constraints, minimax objective, variable bounds, constraint scaling,
mechanism extraction, full LP assembly, solver interface, LPManager, warm-start,
slack tracker, piecewise-linear L2, and end-to-end integration.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import numpy.testing as npt_assert
import pytest
from scipy.sparse import coo_matrix, csr_matrix, issparse

from dp_forge.exceptions import ConfigurationError, InfeasibleSpecError, SolverError
from dp_forge.lp_builder import (
    LPManager,
    SolveStatistics,
    VariableLayout,
    _ApproxDPSlackTracker,
    build_approx_dp_constraints,
    build_and_solve_privacy_lp,
    build_laplace_warm_start,
    build_loss_matrix,
    build_minimax_objective,
    build_output_grid,
    build_privacy_lp,
    build_pure_dp_constraints,
    build_scaled_privacy_lp,
    build_simplex_constraints,
    build_var_map,
    build_variable_bounds,
    estimate_condition_number,
    extract_mechanism_table,
    piecewise_linear_l2_coefficients,
    scale_constraints,
    solve_lp,
    validate_solution,
    detect_degeneracy,
)
from dp_forge.types import (
    AdjacencyRelation,
    LPStruct,
    LossFunction,
    NumericalConfig,
    QuerySpec,
    SolverBackend,
    SynthesisConfig,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_layout() -> VariableLayout:
    return VariableLayout(n=3, k=4)


@pytest.fixture
def layout_with_aux() -> VariableLayout:
    return VariableLayout(n=2, k=3, n_aux=6, aux_labels=("slack_fwd", "slack_bwd"))


@pytest.fixture
def small_spec() -> QuerySpec:
    return QuerySpec(
        query_values=np.array([0.0, 1.0, 2.0]), domain="test",
        sensitivity=1.0, epsilon=1.0, delta=0.0, k=5, loss_fn=LossFunction.L1,
    )


@pytest.fixture
def approx_spec() -> QuerySpec:
    return QuerySpec(
        query_values=np.array([0.0, 1.0, 2.0]), domain="test",
        sensitivity=1.0, epsilon=1.0, delta=0.01, k=5, loss_fn=LossFunction.L1,
    )


@pytest.fixture
def tiny_spec() -> QuerySpec:
    return QuerySpec(
        query_values=np.array([0.0, 1.0]), domain="test",
        sensitivity=1.0, epsilon=1.0, delta=0.0, k=3, loss_fn=LossFunction.L1,
    )


# =========================================================================
# 1. VariableLayout
# =========================================================================

class TestVariableLayout:

    def test_p_index_basic(self, simple_layout: VariableLayout) -> None:
        assert simple_layout.p_index(0, 0) == 0
        assert simple_layout.p_index(0, 3) == 3
        assert simple_layout.p_index(1, 0) == 4
        assert simple_layout.p_index(2, 3) == 11

    @pytest.mark.parametrize("n,k,i,j,expected", [
        (2, 2, 0, 0, 0), (2, 2, 1, 1, 3), (5, 10, 3, 7, 37), (10, 5, 9, 4, 49),
    ])
    def test_p_index_parametrized(self, n, k, i, j, expected) -> None:
        assert VariableLayout(n=n, k=k).p_index(i, j) == expected

    def test_p_indices_row(self, simple_layout: VariableLayout) -> None:
        npt_assert.assert_array_equal(simple_layout.p_indices_row(0), np.arange(0, 4))
        npt_assert.assert_array_equal(simple_layout.p_indices_row(2), np.arange(8, 12))
        assert simple_layout.p_indices_row(0).dtype == np.intp

    def test_t_index(self, simple_layout: VariableLayout) -> None:
        assert simple_layout.t_index == 12

    @pytest.mark.parametrize("n,k,expected", [(1, 2, 2), (2, 3, 6), (5, 10, 50)])
    def test_t_index_parametrized(self, n, k, expected) -> None:
        assert VariableLayout(n=n, k=k).t_index == expected

    def test_aux_start(self, simple_layout: VariableLayout) -> None:
        assert simple_layout.aux_start == 13

    def test_aux_index(self, layout_with_aux: VariableLayout) -> None:
        assert layout_with_aux.aux_index(0) == 7
        assert layout_with_aux.aux_index(5) == 12

    def test_n_vars_no_aux(self, simple_layout: VariableLayout) -> None:
        assert simple_layout.n_vars == 13

    def test_n_vars_with_aux(self, layout_with_aux: VariableLayout) -> None:
        assert layout_with_aux.n_vars == 13

    @pytest.mark.parametrize("n,k,n_aux,expected", [
        (1, 2, 0, 3), (2, 3, 0, 7), (2, 3, 6, 13), (4, 5, 10, 31),
    ])
    def test_n_vars_parametrized(self, n, k, n_aux, expected) -> None:
        assert VariableLayout(n=n, k=k, n_aux=n_aux).n_vars == expected

    def test_frozen(self, simple_layout: VariableLayout) -> None:
        with pytest.raises(AttributeError):
            simple_layout.n = 10  # type: ignore[misc]

    def test_no_index_overlap(self) -> None:
        layout = VariableLayout(n=3, k=4, n_aux=5)
        indices = set()
        for i in range(3):
            for j in range(4):
                idx = layout.p_index(i, j)
                assert idx not in indices
                indices.add(idx)
        assert layout.t_index not in indices
        indices.add(layout.t_index)
        for off in range(5):
            idx = layout.aux_index(off)
            assert idx not in indices
            indices.add(idx)
        assert len(indices) == layout.n_vars

    def test_last_p_before_t(self) -> None:
        for n in range(1, 5):
            for k in range(2, 5):
                lo = VariableLayout(n=n, k=k)
                assert lo.p_index(n - 1, k - 1) == lo.t_index - 1


# =========================================================================
# 2. build_output_grid
# =========================================================================

class TestBuildOutputGrid:

    def test_basic_grid(self) -> None:
        grid = build_output_grid(np.array([0.0, 1.0, 2.0]), k=5)
        assert len(grid) == 5

    def test_grid_sorted(self) -> None:
        grid = build_output_grid(np.array([3.0, 1.0, 2.0]), k=10)
        assert np.all(np.diff(grid) > 0)

    def test_grid_covers_values(self) -> None:
        f = np.array([1.0, 3.0])
        grid = build_output_grid(f, k=5)
        assert grid[0] < f.min() and grid[-1] > f.max()

    def test_grid_padding(self) -> None:
        f = np.array([0.0, 10.0])
        g_def = build_output_grid(f, k=10)
        g_wide = build_output_grid(f, k=10, padding_factor=2.0)
        assert g_wide[0] < g_def[0] and g_wide[-1] > g_def[-1]

    def test_zero_padding(self) -> None:
        grid = build_output_grid(np.array([0.0, 10.0]), k=5, padding_factor=0.0)
        npt_assert.assert_almost_equal(grid[0], 0.0)
        npt_assert.assert_almost_equal(grid[-1], 10.0)

    def test_single_value_degeneracy(self) -> None:
        grid = build_output_grid(np.array([5.0]), k=5)
        assert len(grid) == 5
        npt_assert.assert_almost_equal((grid[0] + grid[-1]) / 2, 5.0)

    def test_k_equals_2(self) -> None:
        grid = build_output_grid(np.array([0.0, 1.0]), k=2)
        assert len(grid) == 2 and grid[0] < grid[1]

    def test_k_less_than_2_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="k must be"):
            build_output_grid(np.array([0.0, 1.0]), k=1)

    def test_identical_values(self) -> None:
        grid = build_output_grid(np.array([3.0, 3.0, 3.0]), k=5)
        npt_assert.assert_almost_equal((grid[0] + grid[-1]) / 2, 3.0)

    @pytest.mark.parametrize("k", [2, 5, 10, 50])
    def test_evenly_spaced(self, k: int) -> None:
        grid = build_output_grid(np.array([0.0, 10.0]), k=k)
        diffs = np.diff(grid)
        npt_assert.assert_allclose(diffs, diffs[0], rtol=1e-12)


# =========================================================================
# 3. build_loss_matrix
# =========================================================================

class TestBuildLossMatrix:

    @pytest.fixture
    def fy(self):
        return np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_l1_values(self, fy) -> None:
        f, y = fy
        L = build_loss_matrix(f, y, LossFunction.L1)
        assert L.shape == (3, 5)
        for i in range(3):
            for j in range(5):
                npt_assert.assert_almost_equal(L[i, j], abs(f[i] - y[j]))
        assert np.all(L >= 0)

    def test_l2_values(self, fy) -> None:
        f, y = fy
        L = build_loss_matrix(f, y, LossFunction.L2)
        for i in range(3):
            for j in range(5):
                npt_assert.assert_almost_equal(L[i, j], (f[i] - y[j]) ** 2)

    def test_linf_equals_l1(self, fy) -> None:
        f, y = fy
        npt_assert.assert_array_almost_equal(
            build_loss_matrix(f, y, LossFunction.LINF),
            build_loss_matrix(f, y, LossFunction.L1),
        )

    def test_custom_loss(self, fy) -> None:
        f, y = fy
        L = build_loss_matrix(f, y, LossFunction.CUSTOM, custom_loss=lambda t, n: (t - n) ** 4)
        for i in range(3):
            for j in range(5):
                npt_assert.assert_almost_equal(L[i, j], (f[i] - y[j]) ** 4)

    def test_custom_none_raises(self, fy) -> None:
        f, y = fy
        with pytest.raises(ConfigurationError, match="custom_loss"):
            build_loss_matrix(f, y, LossFunction.CUSTOM)

    def test_zero_loss_on_diagonal(self) -> None:
        f = np.array([1.0, 2.0, 3.0])
        for lf in [LossFunction.L1, LossFunction.L2, LossFunction.LINF]:
            L = build_loss_matrix(f, f, lf)
            for i in range(3):
                npt_assert.assert_almost_equal(L[i, i], 0.0)

    def test_l2_is_l1_squared(self) -> None:
        f, y = np.array([0.0, 5.0]), np.array([1.0, 3.0, 7.0])
        npt_assert.assert_array_almost_equal(
            build_loss_matrix(f, y, LossFunction.L2),
            build_loss_matrix(f, y, LossFunction.L1) ** 2,
        )

    @pytest.mark.parametrize("n,k", [(1, 2), (5, 5), (10, 20)])
    def test_shape(self, n, k) -> None:
        f = np.arange(n, dtype=np.float64)
        y = np.linspace(-1, n + 1, k)
        assert build_loss_matrix(f, y, LossFunction.L1).shape == (n, k)


# =========================================================================
# 4. build_pure_dp_constraints
# =========================================================================

class TestBuildPureDPConstraints:

    def test_shape_and_rhs(self) -> None:
        layout = VariableLayout(n=3, k=4)
        A, b = build_pure_dp_constraints(0, 1, 4, 1.0, layout)
        assert A.shape == (8, layout.n_vars) and b.shape == (8,)
        npt_assert.assert_array_equal(b, np.zeros(8))

    def test_nnz_per_row(self) -> None:
        layout = VariableLayout(n=3, k=4)
        A, _ = build_pure_dp_constraints(0, 1, 4, 1.0, layout)
        A_d = A.toarray()
        for r in range(A_d.shape[0]):
            assert np.count_nonzero(A_d[r]) == 2

    def test_forward_coefficients(self) -> None:
        eps, e_eps = 1.0, math.exp(1.0)
        layout = VariableLayout(n=3, k=4)
        A_d = build_pure_dp_constraints(0, 1, 4, eps, layout)[0].toarray()
        for j in range(4):
            npt_assert.assert_almost_equal(A_d[2 * j, layout.p_index(0, j)], 1.0)
            npt_assert.assert_almost_equal(A_d[2 * j, layout.p_index(1, j)], -e_eps)

    def test_backward_coefficients(self) -> None:
        eps, e_eps = 1.0, math.exp(1.0)
        layout = VariableLayout(n=3, k=4)
        A_d = build_pure_dp_constraints(0, 1, 4, eps, layout)[0].toarray()
        for j in range(4):
            npt_assert.assert_almost_equal(A_d[2 * j + 1, layout.p_index(1, j)], 1.0)
            npt_assert.assert_almost_equal(A_d[2 * j + 1, layout.p_index(0, j)], -e_eps)

    def test_symmetry(self) -> None:
        layout = VariableLayout(n=3, k=4)
        A01 = build_pure_dp_constraints(0, 1, 4, 1.0, layout)[0].toarray()
        A10 = build_pure_dp_constraints(1, 0, 4, 1.0, layout)[0].toarray()
        for j in range(4):
            npt_assert.assert_array_almost_equal(A01[2 * j], A10[2 * j + 1])

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_neg_coeff_is_minus_exp_eps(self, eps) -> None:
        layout = VariableLayout(n=2, k=3)
        A_d = build_pure_dp_constraints(0, 1, 3, eps, layout)[0].toarray()
        npt_assert.assert_almost_equal(A_d[0, layout.p_index(1, 0)], -math.exp(eps))

    def test_only_touches_pair_columns(self) -> None:
        layout = VariableLayout(n=5, k=3)
        A_d = build_pure_dp_constraints(1, 3, 3, 1.0, layout)[0].toarray()
        valid = {layout.p_index(1, j) for j in range(3)} | {layout.p_index(3, j) for j in range(3)}
        for r in range(A_d.shape[0]):
            for c in range(A_d.shape[1]):
                if A_d[r, c] != 0:
                    assert c in valid

    def test_sparse_format(self) -> None:
        A, _ = build_pure_dp_constraints(0, 1, 3, 1.0, VariableLayout(n=2, k=3))
        assert isinstance(A, coo_matrix)

    @pytest.mark.parametrize("k", [2, 5, 10, 20])
    def test_row_count(self, k) -> None:
        A, b = build_pure_dp_constraints(0, 1, k, 1.0, VariableLayout(n=2, k=k))
        assert A.shape[0] == 2 * k

    def test_total_nnz(self) -> None:
        k = 5
        A, _ = build_pure_dp_constraints(0, 1, k, 1.0, VariableLayout(n=3, k=k))
        assert A.nnz == 4 * k


# =========================================================================
# 5. build_approx_dp_constraints
# =========================================================================

class TestBuildApproxDPConstraints:

    def _make(self, k=4, eps=1.0, delta=0.01):
        layout = VariableLayout(n=2, k=k, n_aux=2 * k)
        A, b = build_approx_dp_constraints(0, 1, k, eps, delta, layout, 0)
        return A, b, layout

    def test_shape(self) -> None:
        A, b, _ = self._make(k=4)
        assert A.shape[0] == 10 and b.shape[0] == 10  # 2*(4+1)

    def test_rhs_delta(self) -> None:
        k, delta = 4, 0.05
        _, b, _ = self._make(k=k, delta=delta)
        npt_assert.assert_almost_equal(b[k], delta)
        npt_assert.assert_almost_equal(b[2 * k + 1], delta)
        for i in range(len(b)):
            if i not in (k, 2 * k + 1):
                npt_assert.assert_almost_equal(b[i], 0.0)

    def test_forward_per_bin(self) -> None:
        k, eps, e_eps = 3, 1.0, math.exp(1.0)
        A, _, layout = self._make(k=k, eps=eps)
        A_d = A.toarray()
        for j in range(k):
            npt_assert.assert_almost_equal(A_d[j, layout.p_index(0, j)], 1.0)
            npt_assert.assert_almost_equal(A_d[j, layout.p_index(1, j)], -e_eps)
            npt_assert.assert_almost_equal(A_d[j, layout.aux_index(j)], -1.0)

    def test_forward_budget(self) -> None:
        k = 3
        A, _, layout = self._make(k=k)
        budget = A.toarray()[k]
        for j in range(k):
            npt_assert.assert_almost_equal(budget[layout.aux_index(j)], 1.0)

    def test_backward_per_bin(self) -> None:
        k, eps, e_eps = 3, 1.0, math.exp(1.0)
        A, _, layout = self._make(k=k, eps=eps)
        A_d = A.toarray()
        for j in range(k):
            row = A_d[k + 1 + j]
            npt_assert.assert_almost_equal(row[layout.p_index(1, j)], 1.0)
            npt_assert.assert_almost_equal(row[layout.p_index(0, j)], -e_eps)
            npt_assert.assert_almost_equal(row[layout.aux_index(k + j)], -1.0)

    def test_backward_budget(self) -> None:
        k = 3
        A, _, layout = self._make(k=k)
        budget = A.toarray()[2 * k + 1]
        for j in range(k):
            npt_assert.assert_almost_equal(budget[layout.aux_index(k + j)], 1.0)

    def test_nnz_per_bin_row(self) -> None:
        A, _, _ = self._make(k=4)
        A_d = A.toarray()
        for j in range(4):
            assert np.count_nonzero(A_d[j]) == 3
            assert np.count_nonzero(A_d[5 + j]) == 3

    def test_total_nnz(self) -> None:
        k = 4
        A, _, _ = self._make(k=k)
        assert A.nnz == 8 * k

    @pytest.mark.parametrize("k", [2, 5, 10])
    def test_row_count_scales(self, k) -> None:
        A, _, _ = self._make(k=k)
        assert A.shape[0] == 2 * (k + 1)


# =========================================================================
# 6. build_simplex_constraints
# =========================================================================

class TestBuildSimplexConstraints:

    def test_shape_and_rhs(self) -> None:
        layout = VariableLayout(n=3, k=4)
        A_eq, b_eq = build_simplex_constraints(3, 4, layout)
        assert A_eq.shape == (3, layout.n_vars)
        npt_assert.assert_array_equal(b_eq, np.ones(3))

    def test_row_sums_correct(self) -> None:
        layout = VariableLayout(n=3, k=4)
        A_d = build_simplex_constraints(3, 4, layout)[0].toarray()
        for i in range(3):
            for j in range(4):
                npt_assert.assert_almost_equal(A_d[i, layout.p_index(i, j)], 1.0)
            npt_assert.assert_almost_equal(A_d[i, layout.t_index], 0.0)

    def test_no_cross_coupling(self) -> None:
        layout = VariableLayout(n=3, k=4)
        A_d = build_simplex_constraints(3, 4, layout)[0].toarray()
        for i in range(3):
            for ip in range(3):
                if ip != i:
                    for j in range(4):
                        assert A_d[i, layout.p_index(ip, j)] == 0.0

    @pytest.mark.parametrize("n,k", [(1, 2), (2, 5), (5, 10)])
    def test_nnz_per_row(self, n, k) -> None:
        layout = VariableLayout(n=n, k=k)
        A_d = build_simplex_constraints(n, k, layout)[0].toarray()
        for i in range(n):
            assert np.count_nonzero(A_d[i]) == k

    def test_total_nnz(self) -> None:
        n, k = 3, 5
        A_eq, _ = build_simplex_constraints(n, k, VariableLayout(n=n, k=k))
        assert A_eq.nnz == n * k

    def test_sparse_format(self) -> None:
        assert isinstance(build_simplex_constraints(2, 3, VariableLayout(n=2, k=3))[0], csr_matrix)


# =========================================================================
# 7. build_minimax_objective
# =========================================================================

class TestBuildMinimaxObjective:

    def test_objective_vector(self) -> None:
        layout = VariableLayout(n=2, k=3)
        c, _, _ = build_minimax_objective(np.ones((2, 3)), 2, 3, layout)
        assert c[layout.t_index] == 1.0 and np.sum(c) == 1.0

    def test_epigraph_rows(self) -> None:
        layout = VariableLayout(n=2, k=3)
        L = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        _, A_obj, b_obj = build_minimax_objective(L, 2, 3, layout)
        A_d = A_obj.toarray()
        npt_assert.assert_array_equal(b_obj, np.zeros(2))
        for i in range(2):
            for j in range(3):
                npt_assert.assert_almost_equal(A_d[i, layout.p_index(i, j)], L[i, j])
            npt_assert.assert_almost_equal(A_d[i, layout.t_index], -1.0)


# =========================================================================
# 8. build_variable_bounds
# =========================================================================

class TestBuildVariableBounds:

    def test_length(self) -> None:
        layout = VariableLayout(n=2, k=3, n_aux=4)
        assert len(build_variable_bounds(layout, 1e-10)) == layout.n_vars

    def test_p_bounds(self) -> None:
        eta = 1e-8
        layout = VariableLayout(n=2, k=3)
        bounds = build_variable_bounds(layout, eta)
        for idx in range(6):
            assert bounds[idx] == (eta, 1.0)

    def test_t_bounds(self) -> None:
        layout = VariableLayout(n=2, k=3)
        assert build_variable_bounds(layout, 1e-10)[layout.t_index] == (0.0, None)

    def test_aux_bounds(self) -> None:
        layout = VariableLayout(n=2, k=3, n_aux=4)
        bounds = build_variable_bounds(layout, 1e-10)
        for off in range(4):
            assert bounds[layout.aux_index(off)] == (0.0, None)


# =========================================================================
# 9. _ApproxDPSlackTracker
# =========================================================================

class TestApproxDPSlackTracker:

    def test_allocations(self) -> None:
        t = _ApproxDPSlackTracker()
        assert t.allocate((0, 1), 4) == 0
        assert t.allocate((1, 2), 4) == 4
        assert t.allocate((0, 1), 4) == 0  # duplicate
        assert t.total_slacks == 8
        assert t.allocations == {(0, 1): 0, (1, 2): 4}

    def test_empty(self) -> None:
        t = _ApproxDPSlackTracker()
        assert t.total_slacks == 0


# =========================================================================
# 10. scale_constraints
# =========================================================================

class TestScaleConstraints:

    def test_empty(self) -> None:
        A_s, _, sf = scale_constraints(csr_matrix((0, 5)), np.empty(0))
        assert A_s.shape == (0, 5) and len(sf) == 0

    def test_coefficients_bounded(self) -> None:
        A = csr_matrix(np.array([[100.0, -200.0], [0.5, -0.3]]))
        A_s, _, _ = scale_constraints(A, np.array([1000.0, 1.0]))
        assert np.all(np.abs(A_s.toarray()) <= 1.0 + 1e-12)

    def test_preserves_feasibility(self) -> None:
        A = csr_matrix(np.array([[2.0, 4.0], [6.0, 8.0]]))
        b = np.array([10.0, 20.0])
        x = np.array([1.0, 1.0])
        A_s, b_s, _ = scale_constraints(A, b)
        assert np.all(A_s.dot(x) <= b_s + 1e-12)

    def test_identity_unchanged(self) -> None:
        A_s, b_s, sf = scale_constraints(csr_matrix(np.eye(3)), np.ones(3))
        npt_assert.assert_array_almost_equal(A_s.toarray(), np.eye(3))
        npt_assert.assert_array_almost_equal(sf, np.ones(3))


# =========================================================================
# 11. estimate_condition_number
# =========================================================================

class TestEstimateConditionNumber:

    def test_identity(self) -> None:
        npt_assert.assert_almost_equal(estimate_condition_number(csr_matrix(np.eye(5))), 1.0, decimal=5)

    def test_ill_conditioned(self) -> None:
        A = np.eye(5); A[0, 0] = 1e10
        assert estimate_condition_number(csr_matrix(A)) > 1e9

    def test_empty(self) -> None:
        assert estimate_condition_number(csr_matrix((0, 5))) == 1.0


# =========================================================================
# 12. extract_mechanism_table
# =========================================================================

class TestExtractMechanismTable:

    def test_shape_and_sum(self) -> None:
        layout = VariableLayout(n=2, k=3)
        x = np.array([0.2, 0.3, 0.5, 0.1, 0.6, 0.3, 1.0])
        P = extract_mechanism_table(x, layout)
        assert P.shape == (2, 3)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_nonneg_after_clamp(self) -> None:
        layout = VariableLayout(n=2, k=3)
        x = np.array([-0.01, 0.5, 0.51, 0.3, 0.3, 0.4, 1.0])
        assert np.all(extract_mechanism_table(x, layout) >= 0)

    def test_renormalization(self) -> None:
        layout = VariableLayout(n=1, k=3)
        x = np.array([0.1, 0.1, 0.1, 1.0])
        npt_assert.assert_allclose(extract_mechanism_table(x, layout).sum(axis=1), 1.0, atol=1e-12)

    def test_does_not_modify_input(self) -> None:
        layout = VariableLayout(n=1, k=3)
        x = np.array([0.2, 0.3, 0.5, 1.0])
        x_copy = x.copy()
        extract_mechanism_table(x, layout)
        npt_assert.assert_array_equal(x, x_copy)


# =========================================================================
# 13. validate_solution & detect_degeneracy
# =========================================================================

class TestValidateSolution:

    def test_valid(self) -> None:
        layout = VariableLayout(n=2, k=2)
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
        A_eq = csr_matrix(np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], dtype=float))
        r = validate_solution(x, layout, csr_matrix((0, 5)), np.empty(0), A_eq, np.ones(2))
        assert r["valid"]

    def test_negative_prob(self) -> None:
        layout = VariableLayout(n=1, k=2)
        r = validate_solution(np.array([-0.1, 1.1, 0.0]), layout, csr_matrix((0, 3)), np.empty(0), None, None)
        assert not r["valid"] and r["min_probability"] < 0


class TestDetectDegeneracy:

    def test_non_degenerate(self) -> None:
        info = detect_degeneracy(csr_matrix(np.eye(3)), np.full(3, 10.0), np.ones(3))
        assert info["n_active"] == 0

    def test_empty(self) -> None:
        info = detect_degeneracy(csr_matrix((0, 5)), np.empty(0), np.ones(5))
        assert not info["is_degenerate"]


# =========================================================================
# 14. piecewise_linear_l2_coefficients
# =========================================================================

class TestPiecewiseLinearL2:

    def test_shapes(self) -> None:
        s, i_ = piecewise_linear_l2_coefficients(5.0, np.linspace(0, 10, 20), 8)
        assert s.shape == (8,) and i_.shape == (8,)

    def test_tangent_touches_quadratic(self) -> None:
        f_val, y_grid, ns = 3.0, np.linspace(0, 6, 20), 10
        slopes, intercepts = piecewise_linear_l2_coefficients(f_val, y_grid, ns)
        tp = np.linspace(y_grid[0], y_grid[-1], ns)
        for s in range(ns):
            npt_assert.assert_almost_equal(slopes[s] * tp[s] + intercepts[s], (f_val - tp[s]) ** 2)

    def test_tangent_below_quadratic(self) -> None:
        f_val, y_grid = 2.0, np.linspace(-1, 5, 50)
        slopes, intercepts = piecewise_linear_l2_coefficients(f_val, y_grid, 20)
        for s in range(len(slopes)):
            assert np.all(slopes[s] * y_grid + intercepts[s] <= (f_val - y_grid) ** 2 + 1e-10)


# =========================================================================
# 15. build_privacy_lp — full assembly
# =========================================================================

class TestBuildPrivacyLP:

    def test_returns_tuple(self, small_spec) -> None:
        lp, layout, meta = build_privacy_lp(small_spec)
        assert isinstance(lp, LPStruct) and isinstance(layout, VariableLayout)

    def test_dimensions(self, small_spec) -> None:
        lp, layout, _ = build_privacy_lp(small_spec)
        assert lp.c.shape == (layout.n_vars,) and len(lp.bounds) == layout.n_vars

    def test_objective_only_at_t(self, small_spec) -> None:
        lp, layout, _ = build_privacy_lp(small_spec)
        c = lp.c.copy(); c[layout.t_index] = 0.0
        assert np.all(c == 0.0)

    def test_pure_dp_metadata(self, small_spec) -> None:
        assert build_privacy_lp(small_spec)[2]["is_pure_dp"]

    def test_approx_dp_metadata(self, approx_spec) -> None:
        assert not build_privacy_lp(approx_spec)[2]["is_pure_dp"]

    def test_eta_min_override(self, small_spec) -> None:
        npt_assert.assert_almost_equal(build_privacy_lp(small_spec, eta_min=1e-5)[2]["eta_min"], 1e-5)

    def test_bounds_eta_min(self, small_spec) -> None:
        lp, layout, meta = build_privacy_lp(small_spec)
        for idx in range(layout.n * layout.k):
            assert lp.bounds[idx][0] == pytest.approx(meta["eta_min"])

    def test_sparse_matrices(self, small_spec) -> None:
        lp, _, _ = build_privacy_lp(small_spec)
        assert issparse(lp.A_ub) and issparse(lp.A_eq)

    def test_pure_dp_constraint_count(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0, 2.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, delta=0.0, k=4, loss_fn=LossFunction.L1)
        _, _, meta = build_privacy_lp(spec)
        assert meta["n_obj_rows"] == 3 and meta["n_dp_rows"] == 16

    def test_approx_dp_constraint_count(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0, 2.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, delta=0.01, k=4, loss_fn=LossFunction.L1)
        assert build_privacy_lp(spec)[2]["n_dp_rows"] == 20

    def test_l2_has_aux(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, k=4, loss_fn=LossFunction.L2)
        assert build_privacy_lp(spec)[1].n_aux >= spec.n

    def test_custom_edges(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0, 2.0, 3.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, k=3, loss_fn=LossFunction.L1)
        assert build_privacy_lp(spec, edges=[(0, 3)])[2]["n_unique_pairs"] == 1


# =========================================================================
# 16. solve_lp
# =========================================================================

class TestSolveLP:

    def _trivial_lp(self):
        n, k = 2, 3
        layout = VariableLayout(n=n, k=k)
        c = np.zeros(layout.n_vars); c[layout.t_index] = 1.0
        A_eq, b_eq = build_simplex_constraints(n, k, layout)
        return LPStruct(c=c, A_ub=csr_matrix((0, layout.n_vars)), b_ub=np.empty(0),
                        A_eq=A_eq, b_eq=b_eq, bounds=build_variable_bounds(layout, 1e-12),
                        var_map=build_var_map(n, k, layout), y_grid=np.linspace(0, 1, k))

    def test_trivial(self) -> None:
        stats = solve_lp(self._trivial_lp(), SolverBackend.HIGHS)
        assert stats.status == "optimal" and len(stats.primal_solution) == 7

    def test_auto_backend(self) -> None:
        assert solve_lp(self._trivial_lp(), SolverBackend.AUTO).status == "optimal"

    def test_full_pure_dp(self, tiny_spec) -> None:
        lp, layout, _ = build_privacy_lp(tiny_spec)
        stats = solve_lp(lp, SolverBackend.HIGHS)
        P = extract_mechanism_table(stats.primal_solution, layout)
        assert P.shape == (2, 3)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(P >= 0)


# =========================================================================
# 17. LPManager
# =========================================================================

class TestLPManager:

    def test_init_empty(self, small_spec) -> None:
        assert LPManager(small_spec, []).n_active_pairs == 0

    def test_add_new(self, small_spec) -> None:
        mgr = LPManager(small_spec, [])
        assert mgr.add_constraints(0, 1) and mgr.n_active_pairs == 1

    def test_add_duplicate(self, small_spec) -> None:
        mgr = LPManager(small_spec, [(0, 1)])
        assert not mgr.add_constraints(0, 1) and mgr.n_active_pairs == 1

    def test_canonical_pair(self, small_spec) -> None:
        mgr = LPManager(small_spec, [])
        mgr.add_constraints(1, 0)
        assert not mgr.add_constraints(0, 1)

    def test_remove(self, small_spec) -> None:
        mgr = LPManager(small_spec, [(0, 1)])
        assert mgr.remove_constraints(0, 1) and mgr.n_active_pairs == 0

    def test_remove_nonexistent(self, small_spec) -> None:
        assert not LPManager(small_spec, []).remove_constraints(0, 1)

    def test_has_pair(self, small_spec) -> None:
        mgr = LPManager(small_spec, [(0, 1)])
        assert mgr.has_pair(0, 1) and mgr.has_pair(1, 0) and not mgr.has_pair(0, 2)

    def test_solve_and_mechanism(self, tiny_spec) -> None:
        mgr = LPManager(tiny_spec, [(0, 1)])
        stats = mgr.solve(solver=SolverBackend.HIGHS)
        assert stats.status == "optimal"
        P = mgr.get_mechanism_table()
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_iteration_counter(self, tiny_spec) -> None:
        mgr = LPManager(tiny_spec, [(0, 1)])
        assert mgr.iteration == 0
        mgr.solve(solver=SolverBackend.HIGHS)
        assert mgr.iteration == 1

    def test_constraint_history(self, small_spec) -> None:
        mgr = LPManager(small_spec, [])
        mgr.add_constraints(0, 1); mgr.add_constraints(1, 2)
        h = mgr.track_constraint_history()
        assert len(h) == 2 and h[0].pair == (0, 1) and h[1].pair == (1, 2)

    def test_warm_start(self, tiny_spec) -> None:
        mgr = LPManager(tiny_spec, [(0, 1)])
        ws = np.zeros(mgr.layout.n_vars)
        for i in range(tiny_spec.n):
            for j in range(tiny_spec.k):
                ws[mgr.layout.p_index(i, j)] = 1.0 / tiny_spec.k
        ws[mgr.layout.t_index] = 1.0
        mgr.warm_start_from_previous(ws)
        assert mgr.solve(solver=SolverBackend.HIGHS).status == "optimal"

    def test_warm_start_wrong_size(self, tiny_spec) -> None:
        mgr = LPManager(tiny_spec, [(0, 1)])
        mgr.warm_start_from_previous(np.ones(100))
        assert mgr.get_last_solution() is None

    def test_no_solve_raises(self, tiny_spec) -> None:
        with pytest.raises(RuntimeError):
            LPManager(tiny_spec, [(0, 1)]).get_mechanism_table()

    def test_get_current_lp(self, tiny_spec) -> None:
        assert isinstance(LPManager(tiny_spec, [(0, 1)]).get_current_lp(), LPStruct)

    def test_validate_solution(self, tiny_spec) -> None:
        mgr = LPManager(tiny_spec, [(0, 1)])
        mgr.solve(solver=SolverBackend.HIGHS)
        assert mgr.validate_last_solution()["valid"]

    def test_numerical_stability(self, tiny_spec) -> None:
        r = LPManager(tiny_spec, [(0, 1)]).check_numerical_stability()
        assert "condition_number" in r and "is_stable" in r


# =========================================================================
# 18. LPManager with approx DP
# =========================================================================

class TestLPManagerApproxDP:

    def test_has_slacks(self, approx_spec) -> None:
        assert LPManager(approx_spec, [(0, 1)]).layout.n_aux > 0

    def test_solve(self, approx_spec) -> None:
        mgr = LPManager(approx_spec, [(0, 1)])
        assert mgr.solve(solver=SolverBackend.HIGHS).status == "optimal"
        npt_assert.assert_allclose(mgr.get_mechanism_table().sum(axis=1), 1.0, atol=1e-6)

    def test_add_pair_increases_slacks(self, approx_spec) -> None:
        mgr = LPManager(approx_spec, [])
        n0 = mgr.layout.n_aux
        mgr.add_constraints(0, 1)
        assert mgr.layout.n_aux == n0 + 2 * approx_spec.k


# =========================================================================
# 19. build_laplace_warm_start
# =========================================================================

class TestBuildLaplaceWarmStart:

    def test_shape(self, tiny_spec) -> None:
        layout = VariableLayout(n=tiny_spec.n, k=tiny_spec.k)
        y = build_output_grid(tiny_spec.query_values, tiny_spec.k)
        assert build_laplace_warm_start(tiny_spec, y, layout).shape == (layout.n_vars,)

    def test_rows_sum_to_one(self, tiny_spec) -> None:
        layout = VariableLayout(n=tiny_spec.n, k=tiny_spec.k)
        y = build_output_grid(tiny_spec.query_values, tiny_spec.k)
        x0 = build_laplace_warm_start(tiny_spec, y, layout)
        for i in range(tiny_spec.n):
            npt_assert.assert_almost_equal(
                sum(x0[layout.p_index(i, j)] for j in range(tiny_spec.k)), 1.0)

    def test_nonneg_and_t_nonneg(self, tiny_spec) -> None:
        layout = VariableLayout(n=tiny_spec.n, k=tiny_spec.k)
        y = build_output_grid(tiny_spec.query_values, tiny_spec.k)
        x0 = build_laplace_warm_start(tiny_spec, y, layout)
        for i in range(tiny_spec.n):
            for j in range(tiny_spec.k):
                assert x0[layout.p_index(i, j)] >= 0
        assert x0[layout.t_index] >= 0

    def test_peak_near_true_value(self) -> None:
        spec = QuerySpec(query_values=np.array([5.0]), domain="t",
                         sensitivity=1.0, epsilon=2.0, k=21, loss_fn=LossFunction.L1)
        layout = VariableLayout(n=1, k=21)
        y = build_output_grid(spec.query_values, 21)
        x0 = build_laplace_warm_start(spec, y, layout)
        probs = np.array([x0[layout.p_index(0, j)] for j in range(21)])
        assert abs(np.argmax(probs) - np.argmin(np.abs(y - 5.0))) <= 1


# =========================================================================
# 20. build_scaled_privacy_lp
# =========================================================================

class TestBuildScaledPrivacyLP:

    def test_scale_factors(self, small_spec) -> None:
        lp, _, sf, _ = build_scaled_privacy_lp(small_spec)
        assert len(sf) == lp.n_ub

    def test_coefficients_bounded(self, small_spec) -> None:
        lp, _, _, _ = build_scaled_privacy_lp(small_spec)
        assert np.all(np.abs(lp.A_ub.toarray()) <= 1.0 + 1e-10)

    def test_still_solvable(self, tiny_spec) -> None:
        lp, _, _, _ = build_scaled_privacy_lp(tiny_spec)
        assert solve_lp(lp, SolverBackend.HIGHS).status == "optimal"


# =========================================================================
# 21. build_and_solve_privacy_lp
# =========================================================================

class TestBuildAndSolve:

    def test_returns_mechanism(self, tiny_spec) -> None:
        P, obj, _, _ = build_and_solve_privacy_lp(tiny_spec, solver=SolverBackend.HIGHS)
        assert P.shape == (2, 3)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert obj >= 0


# =========================================================================
# 22. End-to-end integration
# =========================================================================

class TestEndToEnd:

    def test_counting_pure(self) -> None:
        P, obj, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=1.0, k=5), solver=SolverBackend.HIGHS)
        assert P.shape == (3, 5) and obj >= 0
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_counting_approx(self) -> None:
        P, _, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=1.0, delta=0.01, k=5), solver=SolverBackend.HIGHS)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_approx_better_than_pure(self) -> None:
        _, obj_p, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=1.0, k=10), solver=SolverBackend.HIGHS)
        _, obj_a, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=1.0, delta=0.05, k=10), solver=SolverBackend.HIGHS)
        assert obj_a <= obj_p + 1e-6

    def test_higher_eps_lower_loss(self) -> None:
        _, obj_t, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=0.5, k=10), solver=SolverBackend.HIGHS)
        _, obj_l, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=2.0, k=10), solver=SolverBackend.HIGHS)
        assert obj_l <= obj_t + 1e-6

    def test_pure_dp_satisfied(self) -> None:
        spec = QuerySpec.counting(n=3, epsilon=1.0, k=5)
        P, _, _, _ = build_and_solve_privacy_lp(spec, solver=SolverBackend.HIGHS)
        e_eps = math.exp(spec.epsilon)
        for i in range(2):
            for j in range(5):
                assert P[i, j] / max(P[i + 1, j], 1e-20) <= e_eps + 1e-4

    @pytest.mark.parametrize("loss_fn", [LossFunction.L1, LossFunction.L2, LossFunction.LINF])
    def test_all_losses(self, loss_fn) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, k=4, loss_fn=loss_fn)
        P, obj, _, _ = build_and_solve_privacy_lp(spec, solver=SolverBackend.HIGHS)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(P >= 0) and obj >= 0

    def test_custom_loss(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, k=5, loss_fn=LossFunction.CUSTOM,
                         custom_loss=lambda t, n: (t - n) ** 4)
        P, _, _, _ = build_and_solve_privacy_lp(spec, solver=SolverBackend.HIGHS)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)


# =========================================================================
# 23. CEGIS workflow integration
# =========================================================================

class TestCEGISWorkflow:

    def test_iterative_add(self) -> None:
        spec = QuerySpec.counting(n=4, epsilon=1.0, k=5)
        mgr = LPManager(spec, [])
        mgr.add_constraints(0, 1)
        obj1 = mgr.solve(solver=SolverBackend.HIGHS).objective_value
        mgr.add_constraints(1, 2)
        obj2 = mgr.solve(solver=SolverBackend.HIGHS).objective_value
        assert obj2 >= obj1 - 1e-6
        mgr.add_constraints(2, 3)
        obj3 = mgr.solve(solver=SolverBackend.HIGHS).objective_value
        assert obj3 >= obj2 - 1e-6
        npt_assert.assert_allclose(mgr.get_mechanism_table().sum(axis=1), 1.0, atol=1e-6)

    def test_with_warm_start(self) -> None:
        spec = QuerySpec.counting(n=3, epsilon=1.0, k=5)
        mgr = LPManager(spec, [(0, 1)])
        x0 = build_laplace_warm_start(spec, mgr.y_grid, mgr.layout)
        mgr.warm_start_from_previous(x0)
        assert mgr.solve(solver=SolverBackend.HIGHS).status == "optimal"


# =========================================================================
# 24. Edge cases
# =========================================================================

class TestEdgeCases:

    def test_single_db_no_dp(self) -> None:
        spec = QuerySpec(query_values=np.array([1.0]), domain="t", sensitivity=1.0,
                         epsilon=1.0, k=3, loss_fn=LossFunction.L1,
                         edges=AdjacencyRelation(edges=[], n=1))
        lp, _, _ = build_privacy_lp(spec, edges=[])
        assert solve_lp(lp, SolverBackend.HIGHS).status == "optimal"

    def test_very_small_eps(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0]), domain="t",
                         sensitivity=1.0, epsilon=0.01, k=3, loss_fn=LossFunction.L1)
        lp, layout, _ = build_privacy_lp(spec)
        P = extract_mechanism_table(solve_lp(lp, SolverBackend.HIGHS).primal_solution, layout)
        npt_assert.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_large_eps(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0]), domain="t",
                         sensitivity=1.0, epsilon=10.0, k=5, loss_fn=LossFunction.L1)
        _, obj, _, _ = build_and_solve_privacy_lp(spec, solver=SolverBackend.HIGHS)
        assert obj < 1.0

    def test_complete_adjacency(self) -> None:
        spec = QuerySpec(query_values=np.array([0.0, 1.0, 2.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, k=4, loss_fn=LossFunction.L1,
                         edges=AdjacencyRelation.complete(3))
        assert build_privacy_lp(spec)[2]["n_unique_pairs"] == 3

    def test_negative_query_values(self) -> None:
        spec = QuerySpec(query_values=np.array([-3.0, -1.0, 0.0]), domain="t",
                         sensitivity=1.0, epsilon=1.0, k=5, loss_fn=LossFunction.L1)
        P, _, _, _ = build_and_solve_privacy_lp(spec, solver=SolverBackend.HIGHS)
        assert P.shape == (3, 5)

    def test_sparsity_high(self) -> None:
        spec = QuerySpec.counting(n=10, epsilon=1.0, k=20)
        assert build_privacy_lp(spec)[0].sparsity > 0.9

    def test_eta_min_no_zeros(self) -> None:
        P, _, _, _ = build_and_solve_privacy_lp(
            QuerySpec.counting(n=3, epsilon=1.0, k=5), solver=SolverBackend.HIGHS)
        assert np.all(P > 0)


# =========================================================================
# 25. LPStruct validation
# =========================================================================

class TestLPStructValidation:

    def test_a_ub_mismatch(self) -> None:
        with pytest.raises(ValueError, match="A_ub"):
            LPStruct(c=np.zeros(3), A_ub=csr_matrix(np.zeros((2, 4))), b_ub=np.zeros(2),
                     A_eq=None, b_eq=None, bounds=[(0, 1)] * 3, var_map={}, y_grid=np.zeros(1))

    def test_b_ub_mismatch(self) -> None:
        with pytest.raises(ValueError, match="b_ub"):
            LPStruct(c=np.zeros(3), A_ub=csr_matrix(np.zeros((2, 3))), b_ub=np.zeros(5),
                     A_eq=None, b_eq=None, bounds=[(0, 1)] * 3, var_map={}, y_grid=np.zeros(1))

    def test_bounds_mismatch(self) -> None:
        with pytest.raises(ValueError, match="bounds"):
            LPStruct(c=np.zeros(3), A_ub=csr_matrix(np.zeros((1, 3))), b_ub=np.zeros(1),
                     A_eq=None, b_eq=None, bounds=[(0, 1)] * 5, var_map={}, y_grid=np.zeros(1))


# =========================================================================
# 26. NumericalConfig interaction
# =========================================================================

class TestNumericalConfigInteraction:

    def test_default_eta(self, tiny_spec) -> None:
        _, _, meta = build_privacy_lp(tiny_spec)
        npt_assert.assert_almost_equal(meta["eta_min"], math.exp(-1.0) * 1e-10)

    def test_custom_eta_scale(self, tiny_spec) -> None:
        _, _, meta = build_privacy_lp(tiny_spec, numerical_config=NumericalConfig(eta_min_scale=1e-5))
        npt_assert.assert_almost_equal(meta["eta_min"], math.exp(-1.0) * 1e-5)

    def test_synthesis_config_override(self) -> None:
        spec = QuerySpec.counting(n=2, epsilon=1.0, k=3)
        mgr = LPManager(spec, [(0, 1)], SynthesisConfig(eta_min=1e-3))
        assert mgr.get_current_lp().bounds[0][0] == pytest.approx(1e-3)


# =========================================================================
# 27. SolveStatistics
# =========================================================================

class TestSolveStatistics:

    def test_construction(self) -> None:
        s = SolveStatistics(solver_name="HiGHS", status="optimal", iterations=42,
                            solve_time=1.5, objective_value=3.14, primal_solution=np.zeros(5))
        assert s.solver_name == "HiGHS" and s.iterations == 42

    def test_defaults_none(self) -> None:
        s = SolveStatistics(solver_name="t", status="ok", iterations=0,
                            solve_time=0, objective_value=0, primal_solution=np.zeros(1))
        assert s.dual_solution is None and s.basis_info is None


# =========================================================================
# 28. Parametrized constraint property sweeps
# =========================================================================

class TestParametrizedConstraints:

    @pytest.mark.parametrize("n,k,eps", [(2, 2, 0.1), (3, 3, 0.5), (5, 4, 2.0)])
    def test_pure_dp_rows(self, n, k, eps) -> None:
        A, _ = build_pure_dp_constraints(0, 1, k, eps, VariableLayout(n=n, k=k))
        assert A.shape[0] == 2 * k

    @pytest.mark.parametrize("n,k,eps,delta", [(2, 2, 0.1, 0.01), (3, 3, 0.5, 0.001)])
    def test_approx_dp_rows(self, n, k, eps, delta) -> None:
        layout = VariableLayout(n=n, k=k, n_aux=2 * k)
        A, _ = build_approx_dp_constraints(0, 1, k, eps, delta, layout, 0)
        assert A.shape[0] == 2 * (k + 1)

    @pytest.mark.parametrize("eps", [0.01, 0.1, 0.5, 1.0, 5.0])
    def test_neg_coeff(self, eps) -> None:
        layout = VariableLayout(n=2, k=2)
        A_d = build_pure_dp_constraints(0, 1, 2, eps, layout)[0].toarray()
        npt_assert.assert_almost_equal(A_d[0, layout.p_index(1, 0)], -math.exp(eps))
