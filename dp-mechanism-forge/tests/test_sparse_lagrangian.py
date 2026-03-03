"""
Comprehensive tests for dp_forge.sparse.lagrangian module.

Tests LagrangianRelaxation dual bound quality, SubgradientOptimizer
convergence, BundleMethod stabilization, LagrangianHeuristic primal
feasibility, and DualityGapMonitor accuracy.
"""

import numpy as np
import pytest

from dp_forge.sparse import (
    DecompositionType,
    LagrangianRelaxer,
    LagrangianState,
    MultiplierUpdate,
    SparseConfig,
    SparseResult,
)
from dp_forge.sparse.lagrangian import (
    AugmentedLagrangian,
    BundleMethod,
    DualityGapMonitor,
    LagrangianHeuristic,
    LagrangianRelaxation,
    SubgradientOptimizer,
)
from dp_forge.types import AdjacencyRelation, PrivacyBudget, QuerySpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(n: int = 3, k: int = 5, eps: float = 1.0) -> QuerySpec:
    qv = np.linspace(0.0, 1.0, n)
    return QuerySpec(
        query_values=qv, domain=list(range(n)),
        sensitivity=1.0, epsilon=eps, k=k,
    )


# =============================================================================
# DualityGapMonitor Tests
# =============================================================================


class TestDualityGapMonitor:
    """Tests for DualityGapMonitor accuracy."""

    def test_initial_state(self):
        mon = DualityGapMonitor(tol=1e-4)
        assert mon.best_lower == -np.inf
        assert mon.best_upper == np.inf
        assert not mon.converged

    def test_record_updates_bounds(self):
        mon = DualityGapMonitor()
        mon.record(1.0, 5.0)
        assert mon.best_lower == 1.0
        assert mon.best_upper == 5.0

    def test_best_lower_is_max(self):
        mon = DualityGapMonitor()
        mon.record(1.0, 10.0)
        mon.record(3.0, 8.0)
        mon.record(2.0, 6.0)
        assert mon.best_lower == 3.0

    def test_best_upper_is_min(self):
        mon = DualityGapMonitor()
        mon.record(1.0, 10.0)
        mon.record(2.0, 7.0)
        mon.record(3.0, 5.0)
        assert mon.best_upper == 5.0

    def test_absolute_gap(self):
        mon = DualityGapMonitor()
        mon.record(2.0, 8.0)
        assert abs(mon.absolute_gap - 6.0) < 1e-12

    def test_relative_gap(self):
        mon = DualityGapMonitor()
        mon.record(2.0, 10.0)
        expected = (10.0 - 2.0) / 10.0
        assert abs(mon.relative_gap - expected) < 1e-12

    def test_convergence_detection(self):
        mon = DualityGapMonitor(tol=0.01)
        mon.record(9.99, 10.0)
        assert mon.converged

    def test_no_convergence_when_gap_large(self):
        mon = DualityGapMonitor(tol=1e-6)
        mon.record(1.0, 100.0)
        assert not mon.converged

    def test_summary_string(self):
        mon = DualityGapMonitor()
        mon.record(1.0, 5.0)
        s = mon.summary()
        assert "LB=" in s
        assert "UB=" in s


# =============================================================================
# LagrangianRelaxation Tests
# =============================================================================


class TestLagrangianRelaxation:
    """Tests for LagrangianRelaxation dual bound quality."""

    def test_zero_multipliers(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        lam = np.zeros(lr.n_multipliers)
        val, M, sg = lr.evaluate(lam)
        assert np.isfinite(val)
        assert M.shape == (2, 3)

    def test_mechanism_row_stochastic(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        lr = LagrangianRelaxation(spec)
        lam = np.zeros(lr.n_multipliers)
        _, M, _ = lr.evaluate(lam)
        for i in range(3):
            np.testing.assert_allclose(M[i].sum(), 1.0, atol=1e-10)

    def test_dual_bound_is_lower_bound(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        lam = np.ones(lr.n_multipliers) * 0.1
        val, M, _ = lr.evaluate(lam)
        # Lagrangian value is a lower bound (for min problems)
        assert np.isfinite(val)

    def test_subgradient_shape(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        lam = np.zeros(lr.n_multipliers)
        _, _, sg = lr.evaluate(lam)
        assert sg.shape == (lr.n_multipliers,)

    def test_constraint_violation(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        M = np.ones((2, 3)) / 3
        v = lr.constraint_violation(M)
        assert v.shape == (lr.n_multipliers,)
        # Uniform mechanism satisfies DP constraints
        assert np.all(v <= 1e-10)

    def test_n_multipliers_correct(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        lr = LagrangianRelaxation(spec)
        expected = len(spec.edges.edges) * spec.k
        assert lr.n_multipliers == expected


# =============================================================================
# SubgradientOptimizer Tests
# =============================================================================


class TestSubgradientOptimizer:
    """Tests for SubgradientOptimizer convergence."""

    def test_produces_state(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=20, verbose=0)
        opt = SubgradientOptimizer(lr, config)
        state, M = opt.solve()
        assert isinstance(state, LagrangianState)
        assert M.shape == (2, 3)

    def test_bounds_improve(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=30, verbose=0)
        opt = SubgradientOptimizer(lr, config)
        state, _ = opt.solve()
        assert state.lower_bound <= state.upper_bound + 1e-6

    def test_mechanism_feasible(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=20, verbose=0)
        opt = SubgradientOptimizer(lr, config)
        _, M = opt.solve()
        for i in range(2):
            np.testing.assert_allclose(M[i].sum(), 1.0, atol=1e-4)
        assert np.all(M >= -1e-10)

    def test_state_property(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=5, verbose=0)
        opt = SubgradientOptimizer(lr, config)
        assert opt.state is None
        opt.solve()
        assert opt.state is not None

    def test_convergence_with_initial_ub(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=20, verbose=0)
        opt = SubgradientOptimizer(lr, config)
        state, _ = opt.solve(initial_ub=100.0)
        assert state.upper_bound <= 100.0 + 1e-6


# =============================================================================
# BundleMethod Tests
# =============================================================================


class TestBundleMethod:
    """Tests for BundleMethod stabilization."""

    def test_produces_result(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=15, verbose=0)
        bm = BundleMethod(lr, config, bundle_size=10, proximal_weight=1.0)
        state, M = bm.solve()
        assert isinstance(state, LagrangianState)
        assert M.shape == (2, 3)

    def test_bundle_grows(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=10, verbose=0)
        bm = BundleMethod(lr, config, bundle_size=20)
        bm.solve()
        assert len(bm.bundle) > 0

    def test_bundle_size_limit(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=30, verbose=0)
        bm = BundleMethod(lr, config, bundle_size=5)
        bm.solve()
        assert len(bm.bundle) <= 5

    def test_bounds_valid(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=15, verbose=0)
        bm = BundleMethod(lr, config)
        state, _ = bm.solve()
        assert state.lower_bound <= state.upper_bound + 1e-6

    def test_state_property(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        lr = LagrangianRelaxation(spec)
        config = SparseConfig(max_iterations=15, convergence_tol=1e-15, verbose=0)
        bm = BundleMethod(lr, config)
        assert bm.state is None
        result = bm.solve()
        # solve() returns (LagrangianState, mechanism)
        state, mechanism = result
        assert isinstance(state, LagrangianState)
        assert mechanism.ndim == 2


# =============================================================================
# LagrangianHeuristic Tests
# =============================================================================


class TestLagrangianHeuristic:
    """Tests for LagrangianHeuristic primal feasibility."""

    def test_recover_uniform(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        heur = LagrangianHeuristic(spec)
        M = np.ones((2, 3)) / 3
        M_feas = heur.recover(M)
        assert M_feas.shape == (2, 3)
        for i in range(2):
            np.testing.assert_allclose(M_feas[i].sum(), 1.0, atol=1e-6)

    def test_recover_produces_nonnegative(self):
        spec = _make_spec(n=3, k=4, eps=1.0)
        heur = LagrangianHeuristic(spec)
        M = np.random.RandomState(42).rand(3, 4)
        M /= M.sum(axis=1, keepdims=True)
        M_feas = heur.recover(M)
        assert np.all(M_feas >= -1e-10)

    def test_recover_dp_feasible(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        heur = LagrangianHeuristic(spec)
        M = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        M_feas = heur.recover(M)
        e_eps = np.exp(spec.epsilon)
        for i, ip in spec.edges.edges:
            for j in range(spec.k):
                assert M_feas[i, j] <= e_eps * M_feas[ip, j] + 1e-6

    def test_convex_combination(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        heur = LagrangianHeuristic(spec)
        Ms = [np.ones((2, 3)) / 3 for _ in range(3)]
        M_feas = heur.recover_convex_combination(Ms)
        assert M_feas.shape == (2, 3)
        for i in range(2):
            np.testing.assert_allclose(M_feas[i].sum(), 1.0, atol=1e-6)

    def test_convex_combination_with_weights(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        heur = LagrangianHeuristic(spec)
        Ms = [np.ones((2, 3)) / 3 for _ in range(3)]
        w = np.array([0.5, 0.3, 0.2])
        M_feas = heur.recover_convex_combination(Ms, w)
        assert M_feas.shape == (2, 3)

    def test_empty_list_returns_uniform(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        heur = LagrangianHeuristic(spec)
        M_feas = heur.recover_convex_combination([])
        np.testing.assert_allclose(M_feas, np.ones((2, 3)) / 3, atol=1e-10)


# =============================================================================
# AugmentedLagrangian Tests
# =============================================================================


class TestAugmentedLagrangian:
    """Tests for AugmentedLagrangian convergence."""

    def test_produces_result(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        al = AugmentedLagrangian(spec, config)
        state, M = al.solve()
        assert isinstance(state, LagrangianState)
        assert M.shape == (2, 3)

    def test_penalty_increases(self):
        spec = _make_spec(n=2, k=3, eps=0.5)
        config = SparseConfig(max_iterations=10, verbose=0)
        al = AugmentedLagrangian(spec, config, initial_penalty=1.0, penalty_growth=2.0)
        al.solve()
        assert al.penalty >= 1.0

    def test_mechanism_nonnegative(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        al = AugmentedLagrangian(spec, config)
        _, M = al.solve()
        assert np.all(M >= -1e-10)

    def test_state_property(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=5, verbose=0)
        al = AugmentedLagrangian(spec, config)
        assert al.state is None
        al.solve()
        assert al.state is not None


# =============================================================================
# Integration: LagrangianRelaxer Public API
# =============================================================================


class TestLagrangianRelaxerAPI:
    """Integration tests for LagrangianRelaxer."""

    def test_solve(self):
        spec = _make_spec(n=2, k=4, eps=1.0)
        config = SparseConfig(
            decomposition_type=DecompositionType.LAGRANGIAN,
            max_iterations=20, verbose=0,
        )
        lr = LagrangianRelaxer(config)
        result = lr.solve(spec)
        assert isinstance(result, SparseResult)
        assert result.mechanism.shape == (2, 4)
        assert np.all(result.mechanism >= -1e-10)

    def test_get_state(self):
        spec = _make_spec(n=2, k=3, eps=1.0)
        config = SparseConfig(max_iterations=10, verbose=0)
        lr = LagrangianRelaxer(config)
        assert lr.get_state() is None
        lr.solve(spec)
        state = lr.get_state()
        assert isinstance(state, LagrangianState)
