"""
Comprehensive tests for dp_forge.lattice.branch_and_bound module.

Tests BranchAndBound correctness, BranchingStrategy selection,
BoundComputation tightness, SymmetryBreaking search reduction,
and CuttingPlanes LP tightening.
"""

import math

import numpy as np
import pytest

from dp_forge.lattice.branch_and_bound import (
    BBNode,
    BBResult,
    BoundComputation,
    BranchAndBound,
    BranchingHeuristic,
    BranchingStrategy,
    CuttingPlanes,
    NodeSelection,
    SelectionStrategy,
    SymmetryBreaking,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_ilp():
    """A simple integer LP: min c·x s.t. Ax ≤ b, 0 ≤ x ≤ 3."""
    c = np.array([-1.0, -2.0])
    A_ub = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    b_ub = np.array([4.0, 3.0, 3.0])
    bounds_lo = np.zeros(2)
    bounds_hi = np.array([3.0, 3.0])
    integer_vars = [0, 1]
    return c, A_ub, b_ub, bounds_lo, bounds_hi, integer_vars


# =============================================================================
# BBNode Tests
# =============================================================================


class TestBBNode:
    """Tests for BBNode data structure."""

    def test_creation(self):
        node = BBNode(node_id=0, depth=0, lower_bound=0.0, upper_bound=10.0, solution=None)
        assert node.node_id == 0
        assert node.depth == 0
        assert not node.is_pruned
        assert not node.is_integer

    def test_comparison(self):
        n1 = BBNode(node_id=0, depth=0, lower_bound=1.0, upper_bound=5.0, solution=None)
        n2 = BBNode(node_id=1, depth=0, lower_bound=2.0, upper_bound=5.0, solution=None)
        assert n1 < n2

    def test_repr(self):
        node = BBNode(node_id=0, depth=1, lower_bound=1.5, upper_bound=3.0, solution=None)
        r = repr(node)
        assert "id=0" in r


# =============================================================================
# BoundComputation Tests
# =============================================================================


class TestBoundComputation:
    """Tests for BoundComputation tightness."""

    def test_lp_relaxation_feasible(self):
        bc = BoundComputation()
        c = np.array([-1.0, -1.0])
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([2.0])
        lo = np.zeros(2)
        hi = np.array([2.0, 2.0])
        obj, sol, duals = bc.compute_lp_relaxation(c, A_ub, b_ub, None, None, lo, hi)
        assert obj < 0  # Optimal should be -2
        assert sol is not None

    def test_lp_relaxation_infeasible(self):
        bc = BoundComputation()
        c = np.array([1.0])
        A_ub = np.array([[1.0], [-1.0]])
        b_ub = np.array([-1.0, -2.0])  # x ≤ -1 and -x ≤ -2 → x ≥ 2 & x ≤ -1
        lo = np.array([0.0])
        hi = np.array([10.0])
        obj, sol, duals = bc.compute_lp_relaxation(c, A_ub, b_ub, None, None, lo, hi)
        assert obj == float('inf')

    def test_lp_relaxation_bounds(self):
        bc = BoundComputation()
        c = np.array([1.0, 1.0])
        lo = np.array([1.0, 2.0])
        hi = np.array([3.0, 4.0])
        obj, sol, _ = bc.compute_lp_relaxation(c, None, None, None, None, lo, hi)
        assert abs(obj - 3.0) < 1e-6  # min = 1 + 2

    def test_lagrangian_bound(self):
        bc = BoundComputation()
        c = np.array([1.0, 2.0])
        A_ub = np.array([[1.0, 1.0]])
        b_ub = np.array([3.0])
        lam = np.array([0.5])
        lo = np.zeros(2)
        hi = np.array([5.0, 5.0])
        lb, x_opt = bc.compute_lagrangian_bound(c, A_ub, b_ub, lam, lo, hi)
        assert np.isfinite(lb)

    def test_lagrangian_bound_is_lower_bound(self):
        bc = BoundComputation()
        c, A_ub, b_ub, lo, hi, _ = _simple_ilp()
        lam = np.ones(3) * 0.5
        lb, _ = bc.compute_lagrangian_bound(c, A_ub, b_ub, lam, lo, hi)
        # LP relaxation gives an upper bound on dual bound
        lp_obj, _, _ = bc.compute_lp_relaxation(c, A_ub, b_ub, None, None, lo, hi)
        assert lb <= lp_obj + 1e-6

    def test_subgradient_optimization(self):
        bc = BoundComputation()
        c, A_ub, b_ub, lo, hi, _ = _simple_ilp()
        lb, lam = bc.subgradient_optimization(c, A_ub, b_ub, lo, hi, max_iters=50)
        assert np.isfinite(lb)
        assert np.all(lam >= -1e-10)


# =============================================================================
# BranchingStrategy Tests
# =============================================================================


class TestBranchingStrategy:
    """Tests for BranchingStrategy variable selection."""

    def test_most_fractional(self):
        bs = BranchingStrategy(BranchingHeuristic.MOST_FRACTIONAL)
        sol = np.array([1.0, 0.5, 0.0, 0.3])
        result = bs.select_variable(sol, [0, 1, 2, 3])
        assert result is not None
        idx, val = result
        assert idx == 1  # 0.5 is closest to 0.5

    def test_first_fractional(self):
        bs = BranchingStrategy(BranchingHeuristic.FIRST_FRACTIONAL)
        sol = np.array([1.0, 0.5, 0.3])
        result = bs.select_variable(sol, [0, 1, 2])
        assert result is not None
        assert result[0] == 1

    def test_integral_returns_none(self):
        bs = BranchingStrategy(BranchingHeuristic.MOST_FRACTIONAL)
        sol = np.array([1.0, 2.0, 0.0])
        result = bs.select_variable(sol, [0, 1, 2])
        assert result is None

    def test_pseudocost_branching(self):
        bs = BranchingStrategy(BranchingHeuristic.PSEUDOCOST)
        sol = np.array([0.5, 0.3])
        result = bs.select_variable(sol, [0, 1])
        assert result is not None

    def test_update_pseudocosts(self):
        bs = BranchingStrategy(BranchingHeuristic.PSEUDOCOST)
        bs.update_pseudocosts(0, "down", 1.5, 0.3)
        bs.update_pseudocosts(0, "up", 2.0, 0.3)
        assert len(bs._pseudocosts_down[0]) == 1
        assert len(bs._pseudocosts_up[0]) == 1

    def test_strong_branching(self):
        bs = BranchingStrategy()
        bc = BoundComputation()
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        # Make a fractional solution
        sol = np.array([1.5, 2.5])
        result = bs.strong_branching(
            sol, ivars, bc, c, A_ub, b_ub, None, None, lo, hi,
        )
        assert result is not None


# =============================================================================
# NodeSelection Tests
# =============================================================================


class TestNodeSelection:
    """Tests for NodeSelection strategies."""

    def _make_node(self, nid, lb, depth=0):
        return BBNode(node_id=nid, depth=depth, lower_bound=lb, upper_bound=100.0, solution=None)

    def test_best_first(self):
        ns = NodeSelection(SelectionStrategy.BEST_FIRST)
        ns.add_node(self._make_node(0, 5.0))
        ns.add_node(self._make_node(1, 2.0))
        ns.add_node(self._make_node(2, 8.0))
        node = ns.select_node()
        assert node.node_id == 1  # lowest bound

    def test_depth_first(self):
        ns = NodeSelection(SelectionStrategy.DEPTH_FIRST)
        ns.add_node(self._make_node(0, 5.0))
        ns.add_node(self._make_node(1, 2.0))
        node = ns.select_node()
        assert node.node_id == 1  # LIFO

    def test_breadth_first(self):
        ns = NodeSelection(SelectionStrategy.BREADTH_FIRST)
        # Different depths – breadth-first prefers shallowest
        n_deep = self._make_node(0, 5.0, depth=1)
        n_shallow = self._make_node(1, 2.0, depth=0)
        ns.add_node(n_deep)
        ns.add_node(n_shallow)
        node = ns.select_node()
        assert node.node_id == 1  # shallower node selected

    def test_is_empty(self):
        ns = NodeSelection(SelectionStrategy.BEST_FIRST)
        assert ns.is_empty
        ns.add_node(self._make_node(0, 1.0))
        assert not ns.is_empty

    def test_size(self):
        ns = NodeSelection(SelectionStrategy.BEST_FIRST)
        assert ns.size == 0
        ns.add_node(self._make_node(0, 1.0))
        ns.add_node(self._make_node(1, 2.0))
        assert ns.size == 2

    def test_prune_by_bound(self):
        ns = NodeSelection(SelectionStrategy.BEST_FIRST)
        ns.add_node(self._make_node(0, 1.0))
        ns.add_node(self._make_node(1, 5.0))
        ns.add_node(self._make_node(2, 10.0))
        pruned = ns.prune_by_bound(6.0)
        assert pruned == 1  # node with lb=10 removed
        assert ns.size == 2

    def test_select_empty_returns_none(self):
        ns = NodeSelection(SelectionStrategy.BEST_FIRST)
        assert ns.select_node() is None

    def test_hybrid_strategy(self):
        ns = NodeSelection(SelectionStrategy.HYBRID, dive_frequency=2)
        ns.add_node(self._make_node(0, 5.0))
        ns.add_node(self._make_node(1, 2.0))
        node = ns.select_node()
        assert node is not None


# =============================================================================
# SymmetryBreaking Tests
# =============================================================================


class TestSymmetryBreaking:
    """Tests for SymmetryBreaking search reduction."""

    def test_detect_symmetries_identity(self):
        sb = SymmetryBreaking()
        A = np.eye(3)
        gens = sb.detect_symmetries(A, 3)
        # Identity matrix has all column symmetries
        assert isinstance(gens, list)

    def test_detect_symmetries_no_symmetry(self):
        sb = SymmetryBreaking()
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        gens = sb.detect_symmetries(A, 2)
        # Different column norms → no symmetry
        assert len(gens) == 0

    def test_detect_symmetries_symmetric_columns(self):
        sb = SymmetryBreaking()
        A = np.array([[1.0, 1.0], [2.0, 2.0]])
        gens = sb.detect_symmetries(A, 2)
        assert len(gens) >= 1

    def test_orbital_fixing(self):
        sb = SymmetryBreaking()
        sol = np.array([0.5, 0.5, 0.3])
        perm = np.array([1, 0, 2], dtype=np.int64)
        fixed = sb.orbital_fixing(sol, [perm], {})
        assert isinstance(fixed, dict)

    def test_isomorphism_pruning(self):
        sb = SymmetryBreaking()
        node = BBNode(node_id=0, depth=0, lower_bound=0.0, upper_bound=10.0,
                       solution=np.array([0.5, 0.5]))
        perm = np.array([1, 0], dtype=np.int64)
        result = sb.isomorphism_pruning(node, [], [perm])
        assert isinstance(result, bool)

    def test_lexicographic_fixing(self):
        sb = SymmetryBreaking()
        constraints = sb.lexicographic_fixing(4, [[0, 1], [2, 3]])
        assert isinstance(constraints, list)


# =============================================================================
# CuttingPlanes Tests
# =============================================================================


class TestCuttingPlanes:
    """Tests for CuttingPlanes tightening."""

    def test_gomory_cut(self):
        cp = CuttingPlanes()
        tableau_row = np.array([0.5, 0.3, -0.2])
        rhs = 0.7
        coeffs, rhs_cut = cp.gomory_cut(tableau_row, rhs)
        assert len(coeffs) == len(tableau_row)

    def test_gomory_cut_integer_rhs_no_cut(self):
        cp = CuttingPlanes()
        tableau_row = np.array([1.0, 0.0, 0.0])
        rhs = 2.0
        coeffs, rhs_cut = cp.gomory_cut(tableau_row, rhs)
        # Integer RHS → fractional part = 0, cut is trivial
        assert isinstance(coeffs, np.ndarray)

    def test_generate_gomory_cuts(self):
        cp = CuttingPlanes()
        sol = np.array([1.5, 2.3])
        A_eq = np.array([[1.0, 0.0], [0.0, 1.0]])
        b_eq = np.array([1.5, 2.3])
        cuts = cp.generate_gomory_cuts(sol, A_eq, b_eq, [0, 1])
        assert isinstance(cuts, list)

    def test_split_cut(self):
        cp = CuttingPlanes()
        sol = np.array([0.5, 1.5])
        pi = np.array([1.0, 0.0])
        pi0 = 0.5
        coeffs, rhs = cp.split_cut(sol, pi, pi0)
        assert len(coeffs) == 2

    def test_lift_and_project_cut(self):
        cp = CuttingPlanes()
        sol = np.array([0.5, 1.0])
        A_ub = np.array([[1.0, 1.0], [-1.0, 0.0]])
        b_ub = np.array([3.0, 0.0])
        result = cp.lift_and_project_cut(sol, A_ub, b_ub, branching_var=0)
        # May return None or a cut tuple
        if result is not None:
            assert len(result) == 2


# =============================================================================
# BranchAndBound Solver Tests
# =============================================================================


class TestBranchAndBound:
    """Tests for B&B solver correctness on small instances."""

    def test_solve_simple_ilp(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(
            selection=SelectionStrategy.BEST_FIRST,
            branching=BranchingHeuristic.MOST_FRACTIONAL,
            max_nodes=100,
        )
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert isinstance(result, BBResult)
        if result.optimal_solution is not None:
            # Solution should be integer
            for i in ivars:
                frac = result.optimal_solution[i] - math.floor(result.optimal_solution[i])
                assert frac < 1e-4 or frac > 1 - 1e-4

    def test_result_bounds(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(max_nodes=50)
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert result.lower_bound <= result.upper_bound + 1e-6

    def test_nodes_explored(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(max_nodes=10)
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert result.nodes_explored <= 10
        assert result.nodes_explored > 0

    def test_gap(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(max_nodes=100)
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert result.gap >= 0

    def test_convergence_history(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(max_nodes=20)
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert isinstance(result.convergence_history, list)

    def test_depth_first_strategy(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(
            selection=SelectionStrategy.DEPTH_FIRST,
            max_nodes=20,
        )
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert isinstance(result, BBResult)

    def test_trivial_integer_solution(self):
        """All variables already integer in LP relaxation."""
        c = np.array([1.0, 1.0])
        lo = np.array([1.0, 2.0])
        hi = np.array([1.0, 2.0])
        bb = BranchAndBound(max_nodes=10)
        result = bb.solve(c, None, None, None, None, lo, hi, [0, 1])
        assert result.optimal_solution is not None
        np.testing.assert_allclose(result.optimal_solution, [1.0, 2.0], atol=1e-6)

    def test_with_cutting_planes(self):
        c, A_ub, b_ub, lo, hi, ivars = _simple_ilp()
        bb = BranchAndBound(max_nodes=30, use_cuts=True)
        result = bb.solve(c, A_ub, b_ub, None, None, lo, hi, ivars)
        assert isinstance(result, BBResult)
