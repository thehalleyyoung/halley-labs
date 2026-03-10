"""Tests for causalcert.dag.incremental – incremental d-sep updates."""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.dag.dsep import DSeparationOracle
from causalcert.dag.incremental import IncrementalDSep, AffectedSeparation
from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    NodeSet,
    StructuralEdit,
)

# ── helpers ───────────────────────────────────────────────────────────────

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
# Incremental matches full recomputation
# ═══════════════════════════════════════════════════════════════════════════


class TestIncrementalMatchesFull:
    """After every edit, incremental answers must agree with a fresh oracle."""

    def _check_triples_match(
        self,
        inc: IncrementalDSep,
        triples: list[tuple[int, int, NodeSet]],
    ) -> None:
        fresh = DSeparationOracle(inc.adj)
        for x, y, s in triples:
            assert inc.query(x, y, s) == fresh.is_d_separated(x, y, s), (
                f"Mismatch for ({x}, {y} | {s})"
            )

    def test_add_edge_chain(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset({1})),
            (0, 3, frozenset()),
            (0, 2, frozenset()),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        inc.update(_add(2, 3))
        self._check_triples_match(inc, triples)

    def test_delete_edge_diamond(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 3, frozenset()),
            (1, 2, frozenset({0})),
            (0, 3, frozenset({1, 2})),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        inc.update(_del(0, 1))
        self._check_triples_match(inc, triples)

    def test_reverse_edge(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset()),
            (0, 2, frozenset({1})),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        inc.update(_rev(0, 1))
        self._check_triples_match(inc, triples)

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_random_sequence(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(6, edge_prob=0.3, seed=seed)
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 5, frozenset()),
            (0, 5, frozenset({1})),
            (1, 4, frozenset()),
            (2, 3, frozenset({0})),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        # Try adding an edge
        for u in range(6):
            for v in range(6):
                if u != v and adj[u, v] == 0:
                    test_adj = adj.copy()
                    test_adj[u, v] = 1
                    # Check acyclicity
                    from causalcert.dag.validation import is_dag
                    if is_dag(test_adj):
                        inc_copy = IncrementalDSep(adj.copy(), watched_triples=triples)
                        inc_copy.update(_add(u, v))
                        self._check_triples_match(inc_copy, triples)
                        return  # one success suffices per seed
        # If no valid add found, just pass
        assert True


# ═══════════════════════════════════════════════════════════════════════════
# Affected CI relations
# ═══════════════════════════════════════════════════════════════════════════


class TestAffectedRelations:
    """Only affected CI relations should change."""

    def test_add_irrelevant_edge(self) -> None:
        # 0->1->2 and 3->4->5 (disconnected)
        adj = _adj(6, [(0, 1), (1, 2), (3, 4), (4, 5)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset({1})),  # only touches component {0,1,2}
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        # Adding 3->5 shouldn't affect triples in component {0,1,2}
        affected = inc.update(_add(3, 5))
        for af in affected:
            assert not af.changed, "Disconnected edit should not affect unrelated triple"

    def test_affected_separation_fields(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset()),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        affected = inc.update(_add(0, 2))
        assert len(affected) >= 1
        for af in affected:
            assert isinstance(af, AffectedSeparation)
            assert hasattr(af, "was_separated")
            assert hasattr(af, "is_separated")
            assert hasattr(af, "changed")


# ═══════════════════════════════════════════════════════════════════════════
# Batch updates
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchUpdate:
    def test_batch_equals_sequential(self) -> None:
        adj = _adj(4, [(0, 1), (1, 2), (2, 3)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 3, frozenset()),
            (0, 3, frozenset({1})),
            (0, 2, frozenset({1})),
        ]
        edits = [_del(1, 2), _add(0, 2)]

        # Sequential
        inc_seq = IncrementalDSep(adj.copy(), watched_triples=triples)
        for e in edits:
            inc_seq.update(e)

        # Batch
        inc_batch = IncrementalDSep(adj.copy(), watched_triples=triples)
        inc_batch.batch_update(edits)

        for x, y, s in triples:
            assert inc_seq.query(x, y, s) == inc_batch.query(x, y, s)

    def test_batch_empty(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        inc = IncrementalDSep(adj)
        affected = inc.batch_update([])
        assert len(affected) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Cache consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheConsistency:
    def test_query_before_update(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        inc = IncrementalDSep(adj)
        # Should use fresh computation
        assert inc.query(0, 2, frozenset({1}))

    def test_cache_invalidation(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        inc = IncrementalDSep(adj)
        base_size = inc.cache_size
        inc.query(0, 2, frozenset({1}))
        assert inc.cache_size >= base_size
        inc.invalidate_cache()
        assert inc.cache_size <= base_size + 1  # cache rebuilt to baseline

    def test_cache_size_grows(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset({1})),
            (0, 2, frozenset()),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        size_before = inc.cache_size
        inc.query(0, 1, frozenset())
        # Cache may grow
        assert inc.cache_size >= size_before

    def test_add_remove_watched(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        inc = IncrementalDSep(adj)
        triple = (0, 2, frozenset({1}))
        inc.add_watched_triple(*triple)
        assert inc.n_watched >= 1
        inc.remove_watched_triple(*triple)

    def test_changed_triples_api(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset({1})),
            (0, 2, frozenset()),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)
        changed = inc.changed_triples(_add(0, 2))
        assert isinstance(changed, list)

    def test_adj_property(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        inc = IncrementalDSep(adj)
        np.testing.assert_array_equal(inc.adj, adj)


# ═══════════════════════════════════════════════════════════════════════════
# Multiple sequential edits
# ═══════════════════════════════════════════════════════════════════════════


class TestSequentialEdits:
    def test_three_sequential_edits(self) -> None:
        adj = _adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 4, frozenset()),
            (0, 4, frozenset({2})),
            (1, 3, frozenset()),
        ]
        inc = IncrementalDSep(adj, watched_triples=triples)

        inc.update(_del(2, 3))
        fresh1 = DSeparationOracle(inc.adj)
        for x, y, s in triples:
            assert inc.query(x, y, s) == fresh1.is_d_separated(x, y, s)

        inc.update(_add(0, 3))
        fresh2 = DSeparationOracle(inc.adj)
        for x, y, s in triples:
            assert inc.query(x, y, s) == fresh2.is_d_separated(x, y, s)

    def test_add_then_delete_same_edge(self) -> None:
        adj = _adj(3, [(0, 1)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset()),
        ]
        inc = IncrementalDSep(adj.copy(), watched_triples=triples)
        inc.update(_add(1, 2))
        inc.update(_del(1, 2))
        # Should be back to original
        fresh = DSeparationOracle(inc.adj)
        for x, y, s in triples:
            assert inc.query(x, y, s) == fresh.is_d_separated(x, y, s)

    def test_reverse_then_reverse_back(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 2, frozenset({1})),
            (0, 2, frozenset()),
        ]
        inc = IncrementalDSep(adj.copy(), watched_triples=triples)
        inc.update(_rev(0, 1))
        inc.update(_rev(1, 0))  # reverse back
        fresh = DSeparationOracle(inc.adj)
        for x, y, s in triples:
            assert inc.query(x, y, s) == fresh.is_d_separated(x, y, s)


# ═══════════════════════════════════════════════════════════════════════════
# Stress tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIncrementalStress:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_edits(self, seed: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(6, edge_prob=0.3, seed=seed)
        triples: list[tuple[int, int, NodeSet]] = [
            (0, 5, frozenset({2})),
            (1, 4, frozenset()),
        ]
        inc = IncrementalDSep(adj.copy(), watched_triples=triples)
        for i in range(6):
            for j in range(6):
                if adj[i, j]:
                    inc.update(_del(i, j))
                    break
            else:
                continue
            break
        fresh = DSeparationOracle(inc.adj)
        for x, y, s in triples:
            assert inc.query(x, y, s) == fresh.is_d_separated(x, y, s)
