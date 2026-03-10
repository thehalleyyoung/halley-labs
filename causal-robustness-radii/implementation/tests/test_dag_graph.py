"""Tests for causalcert.dag.graph – CausalDAG creation, edges, queries."""

from __future__ import annotations

import copy
from itertools import combinations

import numpy as np
import pytest

from causalcert.dag.graph import CausalDAG
from causalcert.dag.validation import is_dag, find_cycle
from causalcert.types import EditType, StructuralEdit

# ── helpers ────────────────────────────────────────────────────────────────

_adj = lambda n, edges: (  # noqa: E731
    np.zeros((n, n), dtype=np.int8).__setitem__(
        tuple(zip(*edges)) if edges else ([], []), 1
    )
    or np.zeros((n, n), dtype=np.int8)
)


def _make_adj(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


# ═══════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestConstruction:
    """CausalDAG construction from various inputs."""

    def test_from_edges_basic(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1), (1, 2)])
        assert dag.n_nodes == 3
        assert dag.n_edges == 2
        assert dag.has_edge(0, 1)
        assert dag.has_edge(1, 2)
        assert not dag.has_edge(0, 2)

    def test_from_adjacency_matrix(self) -> None:
        m = _make_adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        dag = CausalDAG.from_adjacency_matrix(m)
        assert dag.n_nodes == 4
        assert dag.n_edges == 4

    def test_empty_factory(self) -> None:
        dag = CausalDAG.empty(5)
        assert dag.n_nodes == 5
        assert dag.n_edges == 0
        assert list(dag.edges()) == []

    def test_custom_node_names(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)], node_names=["A", "B", "C"])
        assert dag.node_names == ["A", "B", "C"]
        assert dag.node_name(0) == "A"
        assert dag.node_id("B") == 1

    def test_default_node_names(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)])
        assert len(dag.node_names) == 3

    def test_validation_rejects_cycle(self) -> None:
        m = _make_adj(3, [(0, 1), (1, 2), (2, 0)])
        with pytest.raises(Exception):
            CausalDAG.from_adjacency_matrix(m, validate=True)

    def test_skip_validation(self) -> None:
        m = _make_adj(3, [(0, 1), (1, 2), (2, 0)])
        dag = CausalDAG.from_adjacency_matrix(m, validate=False)
        assert dag.n_edges == 3  # built but cyclic

    def test_single_node(self) -> None:
        dag = CausalDAG.empty(1)
        assert dag.n_nodes == 1
        assert dag.n_edges == 0
        assert list(dag.nodes()) == [0]

    def test_self_loop_via_matrix_rejected(self) -> None:
        m = np.eye(3, dtype=np.int8)
        with pytest.raises(Exception):
            CausalDAG.from_adjacency_matrix(m, validate=True)


# ═══════════════════════════════════════════════════════════════════════════
# Edge operations
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeOperations:
    """Mutation: add, delete, reverse edges."""

    def test_add_edge(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)])
        dag.add_edge(1, 2)
        assert dag.has_edge(1, 2)
        assert dag.n_edges == 2

    def test_add_edge_creates_cycle_raises(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1), (1, 2)])
        with pytest.raises(Exception):
            dag.add_edge(2, 0)

    def test_delete_edge(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1), (1, 2)])
        dag.delete_edge(0, 1)
        assert not dag.has_edge(0, 1)
        assert dag.n_edges == 1

    def test_delete_absent_edge_raises(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)])
        with pytest.raises(Exception):
            dag.delete_edge(1, 2)

    def test_reverse_edge(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1), (1, 2)])
        dag.reverse_edge(0, 1)
        assert dag.has_edge(1, 0)
        assert not dag.has_edge(0, 1)

    def test_reverse_creates_cycle_raises(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1), (1, 2), (0, 2)])
        with pytest.raises(Exception):
            dag.reverse_edge(0, 2)

    def test_apply_edit_add(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)])
        dag.apply_edit(StructuralEdit(EditType.ADD, 1, 2))
        assert dag.has_edge(1, 2)

    def test_apply_edit_delete(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1), (1, 2)])
        dag.apply_edit(StructuralEdit(EditType.DELETE, 0, 1))
        assert not dag.has_edge(0, 1)

    def test_apply_edit_reverse(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)])
        dag.apply_edit(StructuralEdit(EditType.REVERSE, 0, 1))
        assert dag.has_edge(1, 0)

    def test_contains_dunder(self) -> None:
        dag = CausalDAG.from_edges(3, [(0, 1)])
        assert (0, 1) in dag
        assert (1, 0) not in dag


# ═══════════════════════════════════════════════════════════════════════════
# Queries: parents / children / ancestors / descendants
# ═══════════════════════════════════════════════════════════════════════════


class TestFamilyQueries:
    """parents, children, ancestors, descendants."""

    def test_parents(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert dag.parents(3) == frozenset({1, 2})
        assert dag.parents(0) == frozenset()

    def test_children(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert dag.children(0) == frozenset({1, 2})
        assert dag.children(3) == frozenset()

    def test_ancestors(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        # ancestors of node 3 = {0, 1, 2} (not 3 itself, depends on impl)
        anc = dag.ancestors(3)
        assert 0 in anc
        assert 1 in anc
        assert 2 in anc

    def test_descendants(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        desc = dag.descendants(0)
        assert 1 in desc
        assert 2 in desc
        assert 3 in desc

    def test_is_ancestor(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert dag.is_ancestor(0, 3)
        assert not dag.is_ancestor(3, 0)

    def test_is_descendant(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert dag.is_descendant(3, 0)
        assert not dag.is_descendant(0, 3)

    def test_neighbors(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        nbrs = dag.neighbors(1)
        assert 0 in nbrs  # parent
        assert 3 in nbrs  # child

    def test_roots_and_leaves(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert 0 in dag.roots()
        assert 3 in dag.leaves()

    def test_empty_graph_roots_leaves(self, empty4_adj: np.ndarray) -> None:
        dag = CausalDAG(empty4_adj)
        assert len(dag.roots()) == 4
        assert len(dag.leaves()) == 4


# ═══════════════════════════════════════════════════════════════════════════
# Degree queries
# ═══════════════════════════════════════════════════════════════════════════


class TestDegrees:
    def test_in_out_degree(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert dag.in_degree(0) == 0
        assert dag.out_degree(0) == 2
        assert dag.in_degree(3) == 2
        assert dag.out_degree(3) == 0

    def test_degree(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert dag.degree(0) == 2  # 0 in + 2 out
        assert dag.degree(3) == 2  # 2 in + 0 out

    def test_max_degree_properties(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert dag.max_in_degree >= 0
        assert dag.max_out_degree >= 0
        assert dag.max_degree >= 0

    def test_degree_sequence(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        seq = dag.degree_sequence()
        assert len(seq) == 4

    def test_mean_degree(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert dag.mean_degree() > 0

    def test_density(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        assert 0.0 < dag.density <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Topological sort
# ═══════════════════════════════════════════════════════════════════════════


class TestTopologicalSort:
    def test_basic_chain(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        topo = dag.topological_sort()
        assert topo == [0, 1, 2, 3]

    def test_diamond_order(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        topo = dag.topological_sort()
        assert topo.index(0) < topo.index(1)
        assert topo.index(0) < topo.index(2)
        assert topo.index(1) < topo.index(3)
        assert topo.index(2) < topo.index(3)

    def test_empty_graph_topo(self, empty4_adj: np.ndarray) -> None:
        dag = CausalDAG(empty4_adj)
        topo = dag.topological_sort()
        assert set(topo) == {0, 1, 2, 3}

    @pytest.mark.parametrize("n", [5, 8, 12])
    def test_random_dag_topo_valid(self, n: int) -> None:
        from tests.conftest import random_dag
        adj = random_dag(n, seed=n)
        dag = CausalDAG(adj)
        topo = dag.topological_sort()
        pos = {v: i for i, v in enumerate(topo)}
        for u, v in dag.edges():
            assert pos[u] < pos[v], f"Edge {u}->{v} violates topological order"


# ═══════════════════════════════════════════════════════════════════════════
# Subgraph extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestSubgraph:
    def test_subgraph_basic(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        sub = dag.subgraph(frozenset({0, 1, 3}))
        assert sub.n_nodes == 3
        # 0->1 and 1->3 should be present; 0->2, 2->3 dropped
        assert sub.n_edges >= 1

    def test_ancestral_subgraph(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        sub = dag.ancestral_subgraph(frozenset({2}))
        assert sub.n_nodes <= 4

    def test_subgraph_single_node(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        sub = dag.subgraph(frozenset({0}))
        assert sub.n_nodes == 1
        assert sub.n_edges == 0


# ═══════════════════════════════════════════════════════════════════════════
# Copy / equality / hashing
# ═══════════════════════════════════════════════════════════════════════════


class TestCopyEquality:
    def test_copy(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        dag2 = dag.copy()
        assert dag == dag2
        dag2.add_edge(0, 2)
        assert dag != dag2

    def test_deepcopy(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        dag2 = copy.deepcopy(dag)
        assert dag == dag2

    def test_equality_different_edges(self) -> None:
        d1 = CausalDAG.from_edges(3, [(0, 1)])
        d2 = CausalDAG.from_edges(3, [(0, 2)])
        assert d1 != d2

    def test_hash_equal_for_same_dag(self, chain4_adj: np.ndarray) -> None:
        d1 = CausalDAG(chain4_adj)
        d2 = CausalDAG(chain4_adj.copy())
        assert hash(d1) == hash(d2)

    def test_hash_in_set(self) -> None:
        d1 = CausalDAG.from_edges(3, [(0, 1)])
        d2 = CausalDAG.from_edges(3, [(0, 1)])
        s = {d1, d2}
        assert len(s) == 1

    def test_repr_str(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert "4" in repr(dag) or "CausalDAG" in repr(dag)
        assert str(dag)  # non-empty string


# ═══════════════════════════════════════════════════════════════════════════
# Adjacency matrix conversion
# ═══════════════════════════════════════════════════════════════════════════


class TestConversions:
    def test_to_adjacency_matrix_round_trip(self, diamond4_adj: np.ndarray) -> None:
        dag = CausalDAG(diamond4_adj)
        m = dag.to_adjacency_matrix()
        np.testing.assert_array_equal(m, diamond4_adj)

    def test_to_edge_list(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        el = dag.to_edge_list()
        assert (0, 1) in el
        assert (1, 2) in el
        assert (2, 3) in el
        assert len(el) == 3

    def test_edge_list_round_trip(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2)]
        dag = CausalDAG.from_edges(3, edges)
        el = dag.to_edge_list()
        dag2 = CausalDAG.from_edges(3, el)
        assert dag == dag2


# ═══════════════════════════════════════════════════════════════════════════
# Structural queries
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuralQueries:
    def test_is_connected_chain(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert dag.is_connected()

    def test_disconnected_not_connected(self, disconnected4_adj: np.ndarray) -> None:
        dag = CausalDAG(disconnected4_adj)
        assert not dag.is_connected()

    def test_connected_components(self, disconnected4_adj: np.ndarray) -> None:
        dag = CausalDAG(disconnected4_adj)
        comps = dag.connected_components()
        assert len(comps) == 2

    def test_is_complete(self, complete4_adj: np.ndarray) -> None:
        dag = CausalDAG(complete4_adj)
        assert dag.is_complete()

    def test_chain_not_complete(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert not dag.is_complete()

    def test_has_directed_path(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert dag.has_directed_path(0, 3)
        assert not dag.has_directed_path(3, 0)

    def test_iter_nodes(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert list(dag) == list(dag.nodes())

    def test_len(self, chain4_adj: np.ndarray) -> None:
        dag = CausalDAG(chain4_adj)
        assert len(dag) == 4


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_disconnected_graph_ancestors(self, disconnected4_adj: np.ndarray) -> None:
        dag = CausalDAG(disconnected4_adj)
        anc = dag.ancestors(1)
        assert 0 in anc
        assert 2 not in anc
        assert 3 not in anc

    def test_single_node_operations(self, single_node_adj: np.ndarray) -> None:
        dag = CausalDAG(single_node_adj)
        assert dag.parents(0) == frozenset()
        assert dag.children(0) == frozenset()
        assert dag.topological_sort() == [0]
        assert dag.is_connected()

    def test_self_loop_edit_raises(self) -> None:
        with pytest.raises(ValueError):
            StructuralEdit(EditType.ADD, 0, 0)

    @pytest.mark.parametrize("n", [2, 10, 20])
    def test_empty_graph_properties(self, n: int) -> None:
        dag = CausalDAG.empty(n)
        assert dag.n_edges == 0
        assert dag.density == 0.0 or n <= 1
        assert len(dag.roots()) == n
        assert len(dag.leaves()) == n

    def test_large_random_dag_is_acyclic(self) -> None:
        from tests.conftest import random_dag
        adj = random_dag(50, edge_prob=0.2, seed=99)
        dag = CausalDAG(adj)
        assert is_dag(dag.adj)

    def test_find_cycle_none_for_dag(self, chain4_adj: np.ndarray) -> None:
        assert find_cycle(chain4_adj) is None

    def test_find_cycle_returns_cycle(self) -> None:
        m = _make_adj(3, [(0, 1), (1, 2), (2, 0)])
        cyc = find_cycle(m)
        assert cyc is not None
        assert len(cyc) >= 2
