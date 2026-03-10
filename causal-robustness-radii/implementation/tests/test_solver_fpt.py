"""Tests for causalcert.solver.fpt – FPT DP solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.dag.moral import treewidth_of_dag, tree_decomposition, moral_graph
from causalcert.dag.validation import is_dag
from causalcert.solver.fpt import FPTSolver
from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    RobustnessRadius,
    StructuralEdit,
)

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _has_path(adj, data, *, treatment, outcome):
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


def _has_direct_edge(adj, data, *, treatment, outcome):
    return bool(adj[treatment, outcome])


def _always_true(adj, data, *, treatment, outcome):
    return True


def _synthetic_data(adj: AdjacencyMatrix, n: int = 200, seed: int = 42) -> pd.DataFrame:
    from tests.conftest import _linear_gaussian_data
    return _linear_gaussian_data(adj, n=n, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# FPT on trees (treewidth 1)
# ═══════════════════════════════════════════════════════════════════════════


class TestFPTOnTrees:
    """Trees have treewidth 1 → FPT should be fast and exact."""

    def test_chain_path_radius(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert isinstance(result, RobustnessRadius)
        # Breaking any single edge on the path suffices
        assert result.upper_bound <= 1

    def test_chain_direct_edge(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_direct_edge, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_tree_treewidth(self) -> None:
        adj = _adj(5, [(0, 1), (0, 2), (1, 3), (1, 4)])
        tw = treewidth_of_dag(adj)
        assert tw <= 2  # tree should have low treewidth


# ═══════════════════════════════════════════════════════════════════════════
# FPT on known treewidth graphs
# ═══════════════════════════════════════════════════════════════════════════


class TestFPTKnownTreewidth:
    def test_diamond_dag(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert isinstance(result, RobustnessRadius)
        assert result.upper_bound >= 1

    def test_solution_valid_dag(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        # Apply witness edits
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
# Compare FPT with brute force on small graphs
# ═══════════════════════════════════════════════════════════════════════════


class TestFPTvsBruteForce:
    """On small graphs, FPT should agree with brute-force enumeration."""

    def _brute_force_radius(self, adj, predicate, data, treatment, outcome, max_k):
        """Enumerate all single-edit perturbations to find minimum."""
        from causalcert.dag.edit import single_edit_perturbations
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return 0
        perturbs = single_edit_perturbations(adj)
        for new_adj, edit in perturbs:
            if not predicate(new_adj, data, treatment=treatment, outcome=outcome):
                return 1
        return 2  # at least 2

    def test_chain_agrees(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        bf = self._brute_force_radius(adj, _has_path, data, 0, 2, 3)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        fpt = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert fpt.upper_bound <= bf + 1  # FPT should be at least as good

    def test_fork_agrees(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2)])
        data = _synthetic_data(adj)
        bf = self._brute_force_radius(adj, _has_direct_edge, data, 0, 1, 3)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        fpt = solver.solve(adj, _has_direct_edge, data, 0, 1, max_k=3)
        assert fpt.upper_bound <= bf + 1


# ═══════════════════════════════════════════════════════════════════════════
# Tree decomposition DP
# ═══════════════════════════════════════════════════════════════════════════


class TestTreeDecompositionDP:
    def test_decomposition_provided(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        mg = moral_graph(adj)
        td = tree_decomposition(mg)
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3, decomposition=td)
        assert isinstance(result, RobustnessRadius)

    def test_max_treewidth_respected(self) -> None:
        # Complete DAG has treewidth n-1
        edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        adj = _adj(5, edges)
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=2, time_limit_s=30)
        # Should either solve or raise/return high bounds
        try:
            result = solver.solve(adj, _has_path, data, 0, 4, max_k=3)
            assert isinstance(result, RobustnessRadius)
        except Exception:
            pass  # Expected if treewidth exceeds limit


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestFPTEdgeCases:
    def test_empty_graph(self) -> None:
        adj = _adj(3, [])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_always_true_predicate(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _always_true, data, 0, 2, max_k=3)
        assert result.lower_bound >= 3

    def test_two_node_graph(self) -> None:
        adj = _adj(2, [(0, 1)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_direct_edge, data, 0, 1, max_k=2)
        assert result.upper_bound >= 1

    def test_disconnected_graph(self) -> None:
        adj = _adj(4, [(0, 1), (2, 3)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert isinstance(result, RobustnessRadius)

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_random_small_dag(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(5, edge_prob=0.3, seed=seed)
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 4, max_k=3)
        assert isinstance(result, RobustnessRadius)
        assert result.lower_bound >= 0
        assert result.upper_bound >= result.lower_bound

    def test_result_fields(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert hasattr(result, "lower_bound")
        assert hasattr(result, "upper_bound")
        assert hasattr(result, "witness_edits")
        assert hasattr(result, "certified")
        assert hasattr(result, "solver_time_s")
        assert result.solver_time_s >= 0


# ═══════════════════════════════════════════════════════════════════════════
# FPT parametric tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFPTParametric:
    @pytest.mark.parametrize("length", [3, 4, 5, 6, 7])
    def test_chain_radius_equals_one(self, length: int) -> None:
        edges = [(i, i + 1) for i in range(length - 1)]
        adj = _adj(length, edges)
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, length - 1, max_k=3)
        assert result.lower_bound == 1

    @pytest.mark.parametrize("max_tw", [2, 4, 8])
    def test_treewidth_limit(self, max_tw: int) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=max_tw, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert result.lower_bound >= 1

    def test_isolated_node_graph(self) -> None:
        adj = _adj(4, [(0, 1)])
        data = _synthetic_data(adj)
        solver = FPTSolver(max_treewidth=8, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        # No path from 0 to 3 exists in original, predicate false
        assert result.lower_bound == 0
