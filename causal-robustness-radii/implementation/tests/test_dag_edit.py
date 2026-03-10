"""Tests for causalcert.dag.edit – edit distance, neighbourhood, canonicalization."""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.dag.edit import (
    edit_distance,
    diff_edits,
    apply_edit,
    apply_edits,
    all_single_edits,
    k_neighbourhood,
    canonicalize_edit_sequence,
    edit_path,
    single_edit_perturbations,
)
from causalcert.dag.validation import is_dag
from causalcert.types import AdjacencyMatrix, EditType, StructuralEdit

# ── helper ─────────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _add(u: int, v: int) -> StructuralEdit:
    return StructuralEdit(EditType.ADD, u, v)


def _del(u: int, v: int) -> StructuralEdit:
    return StructuralEdit(EditType.DELETE, u, v)


def _rev(u: int, v: int) -> StructuralEdit:
    return StructuralEdit(EditType.REVERSE, u, v)


# ═══════════════════════════════════════════════════════════════════════════
# Edit distance
# ═══════════════════════════════════════════════════════════════════════════


class TestEditDistance:
    def test_same_graph_zero(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        assert edit_distance(adj, adj) == 0

    def test_one_addition(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        assert edit_distance(adj1, adj2) == 1

    def test_one_deletion(self) -> None:
        adj1 = _adj(3, [(0, 1), (1, 2)])
        adj2 = _adj(3, [(0, 1)])
        assert edit_distance(adj1, adj2) == 1

    def test_one_reversal(self) -> None:
        adj1 = _adj(3, [(0, 1), (1, 2)])
        adj2 = _adj(3, [(1, 0), (1, 2)])
        assert edit_distance(adj1, adj2) == 1

    def test_symmetric(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(1, 2)])
        assert edit_distance(adj1, adj2) == edit_distance(adj2, adj1)

    def test_empty_to_chain(self) -> None:
        adj1 = _adj(3, [])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        assert edit_distance(adj1, adj2) == 2

    def test_complete_to_empty(self) -> None:
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        adj1 = _adj(4, edges)
        adj2 = _adj(4, [])
        assert edit_distance(adj1, adj2) == 6  # 4 choose 2


# ═══════════════════════════════════════════════════════════════════════════
# Diff edits
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffEdits:
    def test_diff_add(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        edits = diff_edits(adj1, adj2)
        assert len(edits) == 1
        assert edits[0].edit_type == EditType.ADD

    def test_diff_delete(self) -> None:
        adj1 = _adj(3, [(0, 1), (1, 2)])
        adj2 = _adj(3, [(0, 1)])
        edits = diff_edits(adj1, adj2)
        assert len(edits) == 1
        assert edits[0].edit_type == EditType.DELETE

    def test_diff_reverse(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(1, 0)])
        edits = diff_edits(adj1, adj2)
        assert len(edits) == 1
        assert edits[0].edit_type == EditType.REVERSE

    def test_diff_empty(self) -> None:
        adj = _adj(3, [(0, 1)])
        edits = diff_edits(adj, adj)
        assert len(edits) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Apply edit
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyEdit:
    def test_apply_add(self) -> None:
        adj = _adj(3, [(0, 1)])
        new = apply_edit(adj, _add(1, 2))
        assert new[1, 2] == 1
        assert new[0, 1] == 1  # original preserved

    def test_apply_delete(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        new = apply_edit(adj, _del(0, 1))
        assert new[0, 1] == 0
        assert new[1, 2] == 1  # other edge preserved

    def test_apply_reverse(self) -> None:
        adj = _adj(3, [(0, 1)])
        new = apply_edit(adj, _rev(0, 1))
        assert new[1, 0] == 1
        assert new[0, 1] == 0

    def test_apply_does_not_mutate(self) -> None:
        adj = _adj(3, [(0, 1)])
        original = adj.copy()
        apply_edit(adj, _add(1, 2))
        np.testing.assert_array_equal(adj, original)

    def test_apply_edits_sequence(self) -> None:
        adj = _adj(3, [(0, 1)])
        edits = [_add(1, 2), _del(0, 1)]
        new = apply_edits(adj, edits)
        assert new[1, 2] == 1
        assert new[0, 1] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Single edits enumeration
# ═══════════════════════════════════════════════════════════════════════════


class TestAllSingleEdits:
    def test_empty_graph(self) -> None:
        adj = _adj(3, [])
        edits = all_single_edits(adj)
        # Can add any of 6 possible edges
        add_edits = [e for e in edits if e.edit_type == EditType.ADD]
        assert len(add_edits) == 6  # 3*2

    def test_complete_dag(self) -> None:
        edges = [(i, j) for i in range(3) for j in range(i + 1, 3)]
        adj = _adj(3, edges)
        edits = all_single_edits(adj)
        del_edits = [e for e in edits if e.edit_type == EditType.DELETE]
        assert len(del_edits) == 3

    def test_includes_reversals(self) -> None:
        adj = _adj(3, [(0, 1)])
        edits = all_single_edits(adj)
        rev_edits = [e for e in edits if e.edit_type == EditType.REVERSE]
        assert len(rev_edits) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# k-neighbourhood enumeration
# ═══════════════════════════════════════════════════════════════════════════


class TestKNeighbourhood:
    def test_k1_chain(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        neighbours = list(k_neighbourhood(adj, 1, acyclic_only=True))
        assert len(neighbours) >= 1
        for new_adj, edits in neighbours:
            assert is_dag(new_adj)
            assert len(edits) <= 1

    def test_k0_is_original(self) -> None:
        adj = _adj(3, [(0, 1)])
        neighbours = list(k_neighbourhood(adj, 0, acyclic_only=True))
        assert len(neighbours) == 1
        np.testing.assert_array_equal(neighbours[0][0], adj)

    def test_k2_includes_k1(self) -> None:
        adj = _adj(3, [(0, 1)])
        k1 = set()
        for new_adj, _ in k_neighbourhood(adj, 1, acyclic_only=True):
            k1.add(new_adj.tobytes())
        k2 = set()
        for new_adj, _ in k_neighbourhood(adj, 2, acyclic_only=True):
            k2.add(new_adj.tobytes())
        assert k1.issubset(k2)

    def test_all_acyclic(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        for new_adj, edits in k_neighbourhood(adj, 1, acyclic_only=True):
            assert is_dag(new_adj), f"Cyclic result from edits: {edits}"


# ═══════════════════════════════════════════════════════════════════════════
# Edit canonicalization
# ═══════════════════════════════════════════════════════════════════════════


class TestCanonicalization:
    def test_canonical_order_deterministic(self) -> None:
        edits1 = [_add(1, 2), _del(0, 1)]
        edits2 = [_del(0, 1), _add(1, 2)]
        c1 = canonicalize_edit_sequence(edits1)
        c2 = canonicalize_edit_sequence(edits2)
        assert c1 == c2

    def test_canonical_single(self) -> None:
        edits = [_add(0, 1)]
        c = canonicalize_edit_sequence(edits)
        assert len(c) == 1

    def test_canonical_empty(self) -> None:
        c = canonicalize_edit_sequence([])
        assert len(c) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Edit path
# ═══════════════════════════════════════════════════════════════════════════


class TestEditPath:
    def test_edit_path_same_graph(self) -> None:
        adj = _adj(3, [(0, 1)])
        path = edit_path(adj, adj)
        assert len(path) == 0

    def test_edit_path_one_edit(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        path = edit_path(adj1, adj2)
        assert len(path) == 1

    def test_edit_path_application(self) -> None:
        adj1 = _adj(3, [(0, 1)])
        adj2 = _adj(3, [(0, 1), (1, 2)])
        path = edit_path(adj1, adj2)
        result = apply_edits(adj1, path)
        np.testing.assert_array_equal(result, adj2)


# ═══════════════════════════════════════════════════════════════════════════
# Single edit perturbations
# ═══════════════════════════════════════════════════════════════════════════


class TestSingleEditPerturbations:
    def test_produces_dags(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        perturbs = single_edit_perturbations(adj)
        for new_adj, edit in perturbs:
            assert is_dag(new_adj)

    def test_each_differs_by_one(self) -> None:
        adj = _adj(3, [(0, 1)])
        perturbs = single_edit_perturbations(adj)
        for new_adj, edit in perturbs:
            assert edit_distance(adj, new_adj) == 1

    def test_empty_graph_perturbations(self) -> None:
        adj = _adj(2, [])
        perturbs = single_edit_perturbations(adj)
        # Can add (0,1) or (1,0)
        assert len(perturbs) == 2
