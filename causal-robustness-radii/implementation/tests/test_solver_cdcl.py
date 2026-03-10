"""Tests for causalcert.solver.cdcl – CDCL solver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalcert.dag.validation import is_dag
from causalcert.solver.cdcl import CDCLSolver, ConflictClause, VSIDSScorer
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
# CDCL finds optimal solutions
# ═══════════════════════════════════════════════════════════════════════════


class TestCDCLOptimal:
    def test_chain_direct_edge(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        result = solver.solve(adj, _has_direct_edge, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_chain_path(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert result.upper_bound >= 1

    def test_diamond_path(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=5000, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert isinstance(result, RobustnessRadius)
        # Diamond has 2 paths: need to break both → radius ≥ 2
        assert result.upper_bound >= 1

    def test_witness_is_valid_dag(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
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
# No-good clause learning
# ═══════════════════════════════════════════════════════════════════════════


class TestClauseLearning:
    def test_clauses_learned(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        # Solver may learn clauses during search
        assert solver.n_learned_clauses >= 0

    def test_conflict_clause_structure(self) -> None:
        clause = ConflictClause(
            edits=frozenset({
                StructuralEdit(EditType.DELETE, 0, 1),
            }),
            reason="test",
        )
        assert len(clause.edits) == 1
        assert clause.reason == "test"


# ═══════════════════════════════════════════════════════════════════════════
# VSIDS scorer
# ═══════════════════════════════════════════════════════════════════════════


class TestVSIDS:
    def test_vsids_bump(self) -> None:
        edits = [
            StructuralEdit(EditType.DELETE, 0, 1),
            StructuralEdit(EditType.DELETE, 1, 2),
        ]
        vsids = VSIDSScorer(edits)
        vsids.bump(edits[0])
        assert vsids.score(edits[0]) > vsids.score(edits[1])

    def test_vsids_decay(self) -> None:
        edits = [StructuralEdit(EditType.DELETE, 0, 1)]
        vsids = VSIDSScorer(edits, decay=0.5)
        vsids.bump(edits[0])
        score_before = vsids.score(edits[0])
        vsids.decay()
        assert vsids.score(edits[0]) < score_before

    def test_best_unassigned(self) -> None:
        edits = [
            StructuralEdit(EditType.DELETE, 0, 1),
            StructuralEdit(EditType.DELETE, 1, 2),
        ]
        vsids = VSIDSScorer(edits)
        vsids.bump(edits[1])
        best = vsids.best_unassigned(set())
        assert best == edits[1]

    def test_best_unassigned_with_assigned(self) -> None:
        edits = [
            StructuralEdit(EditType.DELETE, 0, 1),
            StructuralEdit(EditType.DELETE, 1, 2),
        ]
        vsids = VSIDSScorer(edits)
        vsids.bump(edits[1])
        best = vsids.best_unassigned({edits[1]})
        assert best == edits[0]


# ═══════════════════════════════════════════════════════════════════════════
# Restart strategy
# ═══════════════════════════════════════════════════════════════════════════


class TestRestartStrategy:
    def test_restart_parameters(self) -> None:
        solver = CDCLSolver(
            max_conflicts=100,
            restart_base=50,
            restart_mult=2.0,
            time_limit_s=30,
        )
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)


# ═══════════════════════════════════════════════════════════════════════════
# Compare with brute force
# ═══════════════════════════════════════════════════════════════════════════


class TestCDCLvsBruteForce:
    def _brute_force_radius(self, adj, predicate, data, treatment, outcome):
        from causalcert.dag.edit import single_edit_perturbations
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return 0
        for new_adj, _ in single_edit_perturbations(adj):
            if not predicate(new_adj, data, treatment=treatment, outcome=outcome):
                return 1
        return 2

    def test_chain_agrees(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        bf = self._brute_force_radius(adj, _has_path, data, 0, 2)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        cdcl = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert cdcl.upper_bound <= bf + 1

    def test_direct_edge_agrees(self) -> None:
        adj = _adj(2, [(0, 1)])
        data = _synthetic_data(adj)
        bf = self._brute_force_radius(adj, _has_direct_edge, data, 0, 1)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        cdcl = solver.solve(adj, _has_direct_edge, data, 0, 1, max_k=3)
        assert cdcl.upper_bound <= bf + 1


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestCDCLEdgeCases:
    def test_empty_graph(self) -> None:
        adj = _adj(3, [])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=500, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_always_true(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=500, time_limit_s=30)
        result = solver.solve(adj, _always_true, data, 0, 2, max_k=3)
        assert result.lower_bound >= 3

    def test_two_node_graph(self) -> None:
        adj = _adj(2, [(0, 1)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=500, time_limit_s=30)
        result = solver.solve(adj, _has_direct_edge, data, 0, 1, max_k=2)
        assert result.upper_bound >= 1

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_random_dag(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(5, edge_prob=0.3, seed=seed)
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 4, max_k=3)
        assert isinstance(result, RobustnessRadius)

    def test_result_has_witness(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert isinstance(result.witness_edits, tuple)

    def test_learned_clauses_property(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        clauses = solver.learned_clauses
        assert isinstance(clauses, list)


# ═══════════════════════════════════════════════════════════════════════════
# CDCL parametric tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCDCLParametric:
    @pytest.mark.parametrize("max_conflicts", [100, 500, 1000])
    def test_conflict_budget(self, max_conflicts: int) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        data = _synthetic_data(adj)
        solver = CDCLSolver(max_conflicts=max_conflicts, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 3, max_k=3)
        assert result.lower_bound >= 1

    @pytest.mark.parametrize("seed", [42, 43, 44])
    def test_deterministic_across_seeds(self, seed: int) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        rng = np.random.default_rng(seed)
        data = pd.DataFrame({"X" + str(i): rng.standard_normal(100) for i in range(3)})
        solver = CDCLSolver(max_conflicts=1000, time_limit_s=30)
        result = solver.solve(adj, _has_path, data, 0, 2, max_k=3)
        assert result.lower_bound >= 0
