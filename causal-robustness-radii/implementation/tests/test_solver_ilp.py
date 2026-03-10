"""Tests for causalcert.solver.ilp – ILP solver and LP relaxation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.dag.validation import is_dag
from causalcert.solver.ilp import ILPSolver

try:
    from causalcert.solver.ilp import _MIP_AVAILABLE
except ImportError:
    _MIP_AVAILABLE = False

skip_no_mip = pytest.mark.skipif(not _MIP_AVAILABLE, reason="python-mip not installed")

from causalcert.solver.lp_relaxation import LPRelaxationSolver
from causalcert.solver.constraints import (
    AcyclicityConstraint,
    BudgetConstraint,
    MutualExclusionConstraint,
    compute_edit_cost,
)
from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _always_true(adj, data, *, treatment, outcome):
    """Conclusion always holds."""
    return True


def _always_false(adj, data, *, treatment, outcome):
    """Conclusion never holds."""
    return False


def _has_direct_edge(adj, data, *, treatment, outcome):
    """True if treatment -> outcome edge exists."""
    return bool(adj[treatment, outcome])


def _has_path(adj, data, *, treatment, outcome):
    """True if there is a directed path treatment -> outcome."""
    n = adj.shape[0]
    visited = set()
    stack = [treatment]
    while stack:
        v = stack.pop()
        if v == outcome:
            return True
        if v in visited:
            continue
        visited.add(v)
        for w in range(n):
            if adj[v, w]:
                stack.append(w)
    return False


def _synthetic_data(adj: AdjacencyMatrix, n: int = 200, seed: int = 42) -> pd.DataFrame:
    from tests.conftest import _linear_gaussian_data
    return _linear_gaussian_data(adj, n=n, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# ILP on small DAGs with known radius
# ═══════════════════════════════════════════════════════════════════════════


@skip_no_mip
class TestILPSmallDAGs:
    """Test ILP solver on DAGs where we know the answer."""

    def test_direct_edge_radius_1(self) -> None:
        """0->1: removing the direct edge flips _has_direct_edge."""
        adj = _adj(2, [(0, 1)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_direct_edge, data, 0, 1, max_k=3)
        assert isinstance(result, RobustnessRadius)
        assert result.upper_bound >= 1
        assert result.lower_bound >= 1

    def test_always_true_max_radius(self) -> None:
        """If predicate is always True, radius should be max_k or very large."""
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _always_true, data, 0, 2, max_k=3)
        assert result.lower_bound >= 3  # cannot be flipped

    def test_chain_path_radius(self) -> None:
        """Chain 0->1->2: need to break the directed path."""
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        # Deleting either edge breaks the path: radius=1
        assert result.upper_bound <= 1 or result.upper_bound == 1

    def test_solution_is_acyclic(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_direct_edge, data, 0, 2, max_k=3)
        # Apply witness edits and verify DAG
        new_adj = adj.copy()
        for edit in result.witness_edits:
            if edit.edit_type == EditType.ADD:
                new_adj[edit.source, edit.target] = 1
            elif edit.edit_type == EditType.DELETE:
                new_adj[edit.source, edit.target] = 0
            elif edit.edit_type == EditType.REVERSE:
                new_adj[edit.source, edit.target] = 0
                new_adj[edit.target, edit.source] = 1
        if len(result.witness_edits) > 0:
            assert is_dag(new_adj)


# ═══════════════════════════════════════════════════════════════════════════
# LP relaxation bounds
# ═══════════════════════════════════════════════════════════════════════════


@skip_no_mip
class TestLPRelaxation:
    def test_lp_lower_bound(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = LPRelaxationSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert result.lower_bound >= 0

    def test_lp_bound_leq_ilp(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        lp = LPRelaxationSolver(time_limit_s=30, verbose=False)
        ilp = ILPSolver(time_limit_s=30, verbose=False)
        lp_result = lp.solve(adj, _has_path, data, 0, 2, max_k=3)
        ilp_result = ilp.solve(adj, _has_path, data, 0, 2, max_k=3)
        # LP relaxation lower bound <= ILP optimal
        assert lp_result.lower_bound <= ilp_result.upper_bound

    def test_fractional_solution(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        solver = LPRelaxationSolver(time_limit_s=30, verbose=False)
        frac = solver.fractional_solution(adj, max_k=3)
        assert isinstance(frac, dict)
        for key, val in frac.items():
            assert 0.0 <= val <= 1.0 + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Constraint generation
# ═══════════════════════════════════════════════════════════════════════════


class TestConstraints:
    def test_edit_cost_identity(self) -> None:
        adj = _adj(3, [(0, 1)])
        assert compute_edit_cost(adj, adj) == 0

    def test_edit_cost_one_edge(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        assert compute_edit_cost(adj1, adj2) == 1

    def test_edit_cost_symmetric(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(1, 2)])
        assert compute_edit_cost(adj1, adj2) == compute_edit_cost(adj2, adj1)


# ═══════════════════════════════════════════════════════════════════════════
# ILP warm start
# ═══════════════════════════════════════════════════════════════════════════


@skip_no_mip
class TestILPWarmStart:
    def test_warm_start_from_adj(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        solver = ILPSolver(time_limit_s=30, verbose=False)
        solver.set_warm_start_from_adj(adj)
        data = _synthetic_data(adj)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)


# ═══════════════════════════════════════════════════════════════════════════
# ILP incremental solving
# ═══════════════════════════════════════════════════════════════════════════


@skip_no_mip
class TestILPIncremental:
    def test_incremental_solve(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve_incremental(
            adj, _has_path, data, 0, 2, k_values=[1, 2, 3]
        )
        assert isinstance(result, RobustnessRadius)

    def test_incremental_matches_direct(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        r_direct = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        r_incr = solver.solve_incremental(adj, _has_path, data, 0, 2, k_values=[1, 2, 3])
        assert r_direct.lower_bound == r_incr.lower_bound


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


@skip_no_mip
class TestILPEdgeCases:
    def test_empty_graph(self) -> None:
        adj = _adj(3, [])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_single_node(self) -> None:
        adj = _adj(2, [(0, 1)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_direct_edge, data, 0, 1, max_k=2)
        assert result.upper_bound >= 1

    def test_disconnected_treatment_outcome(self) -> None:
        adj = _adj(4, [(0, 1), (2, 3)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert isinstance(result, RobustnessRadius)

    @pytest.mark.parametrize("n_nodes", [3, 4, 5])
    def test_complete_dag(self, n_nodes: int) -> None:
        edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
        adj = _adj(n_nodes, edges)
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, n_nodes - 1, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_result_fields(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert hasattr(result, "lower_bound")
        assert hasattr(result, "upper_bound")
        assert hasattr(result, "witness_edits")
        assert hasattr(result, "solver_time_s")
        assert result.solver_time_s >= 0
        assert result.gap >= 0

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_small_dag(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(4, edge_prob=0.4, seed=seed)
        data = _synthetic_data(adj)
        solver = ILPSolver(time_limit_s=30, verbose=False)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert isinstance(result, RobustnessRadius)
        assert result.lower_bound >= 0


# ═══════════════════════════════════════════════════════════════════════════
# LP relaxation edge importance
# ═══════════════════════════════════════════════════════════════════════════


@skip_no_mip
class TestLPEdgeImportance:
    def test_edge_importance(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        solver = LPRelaxationSolver(time_limit_s=30, verbose=False)
        importance = solver.edge_importance(adj)
        assert isinstance(importance, dict)

    def test_integrality_gap(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        solver = LPRelaxationSolver(time_limit_s=30, verbose=False)
        gap = solver.integrality_gap(adj, integer_opt=1, max_k=3)
        assert gap >= 0.0
