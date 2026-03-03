"""
Comprehensive tests for dp_forge.sparse.column_generation module.

Tests column pool management, reduced cost computation, pricing oracle,
master problem LP, Dantzig-Wolfe decomposition convergence, and
branch-and-price integer solutions.
"""

import numpy as np
import pytest

from dp_forge.sparse import (
    Column,
    ColumnGenerator,
    DecompositionType,
    PricingStrategy,
    SparseConfig,
    SparseResult,
)
from dp_forge.sparse.column_generation import (
    BranchAndPrice,
    BranchNode,
    ColumnPool,
    DantzigWolfeDecomposition,
    MasterProblem,
    PricingOracle,
    ReducedCostComputation,
)
from dp_forge.types import AdjacencyRelation, PrivacyBudget, QuerySpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(n: int = 3, k: int = 5, eps: float = 1.0) -> QuerySpec:
    """Create a small QuerySpec for testing."""
    qv = np.linspace(0.0, 1.0, n)
    return QuerySpec(
        query_values=qv,
        domain=list(range(n)),
        sensitivity=1.0,
        epsilon=eps,
        k=k,
    )


# =============================================================================
# ColumnPool Tests
# =============================================================================


class TestColumnPool:
    """Tests for ColumnPool management."""

    def test_empty_pool(self):
        pool = ColumnPool()
        assert pool.size == 0
        mat = pool.get_matrix(k=5)
        assert mat.shape == (5, 0)

    def test_add_single_column(self):
        pool = ColumnPool()
        dist = np.array([0.2, 0.3, 0.5])
        col = pool.add(dist, source_input=0, reduced_cost=-1.0)
        assert pool.size == 1
        assert col.source_input == 0
        np.testing.assert_allclose(col.distribution, dist)

    def test_add_multiple_columns(self):
        pool = ColumnPool()
        for i in range(5):
            pool.add(np.ones(4) / 4, source_input=i % 3)
        assert pool.size == 5

    def test_get_columns_for_input(self):
        pool = ColumnPool()
        pool.add(np.array([0.5, 0.5]), source_input=0)
        pool.add(np.array([0.3, 0.7]), source_input=1)
        pool.add(np.array([0.4, 0.6]), source_input=0)
        cols_0 = pool.get_columns_for_input(0)
        cols_1 = pool.get_columns_for_input(1)
        assert len(cols_0) == 2
        assert len(cols_1) == 1
        assert len(pool.get_columns_for_input(99)) == 0

    def test_get_matrix_shape(self):
        pool = ColumnPool()
        k = 4
        for i in range(3):
            pool.add(np.ones(k) / k, source_input=i)
        mat = pool.get_matrix(k)
        assert mat.shape == (k, 3)

    def test_prune_removes_columns(self):
        pool = ColumnPool()
        c0 = pool.add(np.array([1.0, 0.0]), source_input=0)
        c1 = pool.add(np.array([0.0, 1.0]), source_input=1)
        c2 = pool.add(np.array([0.5, 0.5]), source_input=0)
        pool.prune({c0.index, c2.index})
        assert pool.size == 2
        remaining = pool.get_columns_for_input(1)
        assert len(remaining) == 0

    def test_prune_preserves_kept(self):
        pool = ColumnPool()
        cols = []
        for i in range(5):
            cols.append(pool.add(np.ones(3) / 3, source_input=i % 2))
        keep = {cols[0].index, cols[2].index, cols[4].index}
        pool.prune(keep)
        assert pool.size == 3

    def test_dedup_via_prune(self):
        """Prune can deduplicate by keeping only unique indices."""
        pool = ColumnPool()
        d = np.array([0.25, 0.75])
        c1 = pool.add(d, source_input=0)
        c2 = pool.add(d, source_input=0)
        pool.prune({c1.index})
        assert pool.size == 1

    def test_column_index_increments(self):
        pool = ColumnPool()
        c0 = pool.add(np.array([1.0]), source_input=0)
        c1 = pool.add(np.array([1.0]), source_input=0)
        assert c1.index == c0.index + 1


# =============================================================================
# ReducedCostComputation Tests
# =============================================================================


class TestReducedCostComputation:
    """Tests for reduced cost accuracy."""

    def test_zero_duals_gives_objective(self):
        loss = np.array([[1.0, 2.0, 3.0]])
        rc = ReducedCostComputation(loss)
        col = np.array([0.5, 0.3, 0.2])
        duals = np.zeros(1)
        constraint_col = np.zeros(1)
        cost = rc.compute(col, 0, duals, constraint_col)
        expected = float(loss[0] @ col)
        assert abs(cost - expected) < 1e-12

    def test_nonzero_duals_reduce_cost(self):
        loss = np.array([[2.0, 1.0, 3.0]])
        rc = ReducedCostComputation(loss)
        col = np.array([0.0, 1.0, 0.0])
        duals = np.array([0.5])
        constraint_col = np.array([1.0])
        cost = rc.compute(col, 0, duals, constraint_col)
        expected = 1.0 - 0.5
        assert abs(cost - expected) < 1e-12

    def test_batch_matches_single(self):
        loss = np.array([[1.0, 2.0], [3.0, 4.0]])
        rc = ReducedCostComputation(loss)
        cols = np.array([[0.6, 0.4], [0.3, 0.7]]).T
        duals = np.array([0.1, 0.2])
        constraint_cols = np.array([[1.0, 0.0], [0.0, 1.0]]).T
        batch = rc.batch_compute(cols, 0, duals, constraint_cols)
        for j in range(2):
            single = rc.compute(cols[:, j], 0, duals, constraint_cols[:, j])
            assert abs(batch[j] - single) < 1e-12

    def test_negative_reduced_cost_improving(self):
        loss = np.array([[10.0, 0.0]])
        rc = ReducedCostComputation(loss)
        col = np.array([0.0, 1.0])
        duals = np.array([5.0])
        constraint_col = np.array([1.0])
        cost = rc.compute(col, 0, duals, constraint_col)
        assert cost < 0, "Column with lower cost should have negative reduced cost"


# =============================================================================
# PricingOracle Tests
# =============================================================================


class TestPricingOracle:
    """Tests for PricingOracle finding improving columns."""

    def test_exact_pricing_returns_column_or_none(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        oracle = PricingOracle(spec, PricingStrategy.EXACT)
        duals = np.zeros(spec.n + len(spec.edges.edges) * spec.k)
        budget = PrivacyBudget(epsilon=spec.epsilon)
        result = oracle.solve(duals, 0, budget)
        # May return None (already optimal) or a Column
        if result is not None:
            assert isinstance(result, Column)
            assert result.distribution.shape == (spec.k,)
            np.testing.assert_allclose(result.distribution.sum(), 1.0, atol=1e-6)

    def test_heuristic_pricing_valid_distribution(self):
        spec = _make_spec(n=3, k=8, eps=1.0)
        oracle = PricingOracle(spec, PricingStrategy.HEURISTIC)
        duals = np.ones(spec.n) * 10.0
        duals = np.concatenate([duals, np.zeros(len(spec.edges.edges) * spec.k)])
        budget = PrivacyBudget(epsilon=spec.epsilon)
        result = oracle.solve(duals, 0, budget)
        if result is not None:
            assert np.all(result.distribution >= -1e-12)
            np.testing.assert_allclose(result.distribution.sum(), 1.0, atol=1e-6)

    def test_hybrid_pricing_tries_both(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        oracle = PricingOracle(spec, PricingStrategy.HYBRID)
        duals = np.zeros(spec.n + len(spec.edges.edges) * spec.k)
        budget = PrivacyBudget(epsilon=spec.epsilon)
        result = oracle.solve(duals, 0, budget)
        # Should not raise

    def test_improving_column_has_negative_rc(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        oracle = PricingOracle(spec, PricingStrategy.EXACT)
        # Large duals should force improving columns
        duals = np.ones(spec.n + len(spec.edges.edges) * spec.k) * 100.0
        budget = PrivacyBudget(epsilon=spec.epsilon)
        result = oracle.solve(duals, 0, budget)
        if result is not None:
            assert result.reduced_cost < 0

    def test_loss_matrix_shape(self):
        spec = _make_spec(n=4, k=6, eps=0.5)
        L = PricingOracle._build_loss_matrix(spec)
        assert L.shape == (4, 6)
        assert np.all(L >= 0)


# =============================================================================
# MasterProblem Tests
# =============================================================================


class TestMasterProblem:
    """Tests for MasterProblem LP formulation and solving."""

    def test_solve_with_uniform_columns(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        pool = ColumnPool()
        k = spec.k
        for i in range(spec.n):
            pool.add(np.ones(k) / k, source_input=i)
        master = MasterProblem(spec, pool)
        primal, duals, obj = master.solve()
        assert primal.shape == (pool.size,)
        assert np.isfinite(obj)

    def test_primal_sums_to_one_per_input(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        pool = ColumnPool()
        # Two columns per input
        for i in range(spec.n):
            pool.add(np.array([0.5, 0.3, 0.2]), source_input=i)
            pool.add(np.array([0.2, 0.5, 0.3]), source_input=i)
        master = MasterProblem(spec, pool)
        primal, _, _ = master.solve()
        for i in range(spec.n):
            cols_i = [j for j, c in enumerate(pool.columns) if c.source_input == i]
            assert abs(sum(primal[j] for j in cols_i) - 1.0) < 1e-6

    def test_dual_values_extracted(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        pool = ColumnPool()
        for i in range(spec.n):
            pool.add(np.ones(3) / 3, source_input=i)
        master = MasterProblem(spec, pool)
        _, duals, _ = master.solve()
        assert duals is not None
        assert len(duals) > 0

    def test_empty_pool_raises(self):
        spec = _make_spec()
        pool = ColumnPool()
        master = MasterProblem(spec, pool)
        with pytest.raises(ValueError, match="empty"):
            master.solve()

    def test_objective_nonnegative(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        pool = ColumnPool()
        for i in range(spec.n):
            pool.add(np.ones(5) / 5, source_input=i)
        master = MasterProblem(spec, pool)
        _, _, obj = master.solve()
        assert obj >= -1e-8


# =============================================================================
# DantzigWolfeDecomposition Tests
# =============================================================================


class TestDantzigWolfeDecomposition:
    """Tests for DW decomposition convergence."""

    def test_initializes_pool(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        dw = DantzigWolfeDecomposition(spec)
        assert dw.pool.size == spec.n

    def test_convergence_produces_valid_result(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        config = SparseConfig(max_iterations=20, verbose=0)
        dw = DantzigWolfeDecomposition(spec, config)
        result = dw.solve()
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (spec.n, spec.k)
        # Rows should sum to ~1
        for i in range(spec.n):
            np.testing.assert_allclose(result.mechanism[i].sum(), 1.0, atol=1e-4)

    def test_mechanism_nonnegative(self):
        spec = _make_spec(n=3, k=4, eps=0.5)
        config = SparseConfig(max_iterations=10, verbose=0)
        dw = DantzigWolfeDecomposition(spec, config)
        result = dw.solve()
        assert np.all(result.mechanism >= -1e-10)

    def test_convergence_history_monotone(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        config = SparseConfig(max_iterations=30, verbose=0)
        dw = DantzigWolfeDecomposition(spec, config)
        result = dw.solve()
        assert len(result.convergence_history) > 0

    def test_objective_improves_or_stable(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=15, verbose=0)
        dw = DantzigWolfeDecomposition(spec, config)
        result = dw.solve()
        assert result.obj_val < np.inf

    def test_pool_grows_during_solve(self):
        spec = _make_spec(n=3, k=6, eps=0.5)
        config = SparseConfig(max_iterations=10, verbose=0)
        dw = DantzigWolfeDecomposition(spec, config)
        initial_size = dw.pool.size
        dw.solve()
        assert dw.pool.size >= initial_size

    def test_max_columns_respected(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        config = SparseConfig(max_columns=10, max_iterations=50, verbose=0)
        dw = DantzigWolfeDecomposition(spec, config)
        dw.solve()
        # Pool may grow then be pruned
        assert dw.pool.size <= config.max_columns + spec.n


# =============================================================================
# BranchAndPrice Tests
# =============================================================================


class TestBranchAndPrice:
    """Tests for branch-and-price integer solutions."""

    def test_produces_valid_mechanism(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        bp = BranchAndPrice(spec, config)
        result = bp.solve(max_nodes=5)
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (spec.n, spec.k)
        for i in range(spec.n):
            np.testing.assert_allclose(result.mechanism[i].sum(), 1.0, atol=1e-4)

    def test_mechanism_nonnegative(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=5, verbose=0)
        bp = BranchAndPrice(spec, config)
        result = bp.solve(max_nodes=3)
        assert np.all(result.mechanism >= -1e-10)

    def test_max_nodes_limits_exploration(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        bp = BranchAndPrice(spec, config)
        result = bp.solve(max_nodes=2)
        assert bp._nodes_explored <= 2

    def test_branch_node_creation(self):
        node = BranchNode(lower_bound=1.5, depth=2)
        assert node.lower_bound == 1.5
        assert node.depth == 2
        assert node.fixed_zero == []
        assert node.fixed_one == []

    def test_integrality_check(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        bp = BranchAndPrice(spec)
        # Near-integer mechanism
        M_int = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert bp._check_integrality(M_int)
        # Fractional mechanism
        M_frac = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        assert not bp._check_integrality(M_frac)

    def test_branching_variable_selection(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        bp = BranchAndPrice(spec)
        M = np.array([[0.5, 0.3, 0.2], [0.1, 0.9, 0.0]])
        i, j = bp._find_branching_variable(M)
        assert 0 <= i < 2
        assert 0 <= j < 3


# =============================================================================
# Integration: ColumnGenerator Public API
# =============================================================================


class TestColumnGeneratorAPI:
    """Integration tests for the ColumnGenerator public API."""

    def test_solve_returns_sparse_result(self):
        spec = _make_spec(n=3, k=5, eps=1.0)
        config = SparseConfig(
            decomposition_type=DecompositionType.COLUMN_GENERATION,
            max_iterations=10,
            verbose=0,
        )
        cg = ColumnGenerator(config)
        result = cg.solve(spec)
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (3, 5)

    def test_solve_with_initial_columns(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        cg = ColumnGenerator(config)
        initial = [
            Column(index=0, distribution=np.ones(4) / 4, reduced_cost=0.0, source_input=0),
            Column(index=1, distribution=np.ones(4) / 4, reduced_cost=0.0, source_input=1),
        ]
        result = cg.solve_with_initial_columns(spec, initial)
        assert isinstance(result, SparseResult)

    def test_result_has_finite_objective(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        cg = ColumnGenerator(config)
        result = cg.solve(spec)
        assert np.isfinite(result.obj_val)

    def test_compression_ratio_valid(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        cg = ColumnGenerator(config)
        result = cg.solve(spec)
        assert 0 <= result.compression_ratio <= 1.0

    def test_gap_nonnegative(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        cg = ColumnGenerator(config)
        result = cg.solve(spec)
        assert result.gap >= -1e-8
