"""Tests for causalcert.dag.moral – moral graph, treewidth, tree decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from causalcert.dag.moral import (
    moral_graph,
    treewidth,
    treewidth_of_dag,
    tree_decomposition,
    TreeDecomposition,
    is_chordal,
    perfect_elimination_ordering,
    chordal_completion,
    maximal_cliques,
    junction_tree,
    JunctionTree,
    treewidth_feasibility,
)
from causalcert.types import AdjacencyMatrix, NodeSet

# ── helper ─────────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _undirected(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
        a[v, u] = 1
    return a


# ═══════════════════════════════════════════════════════════════════════════
# Moral graph construction
# ═══════════════════════════════════════════════════════════════════════════


class TestMoralGraph:
    """Moralization adds edges between co-parents and drops directions."""

    def test_chain_moral(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        mg = moral_graph(adj)
        # Chain: no co-parents, just undirected skeleton
        assert mg[0, 1] == 1 and mg[1, 0] == 1
        assert mg[1, 2] == 1 and mg[2, 1] == 1
        assert mg[0, 2] == 0

    def test_collider_moral(self) -> None:
        adj = _adj(3, [(0, 2), (1, 2)])
        mg = moral_graph(adj)
        # Co-parents 0 and 1 should be married
        assert mg[0, 1] == 1 and mg[1, 0] == 1

    def test_fork_moral(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2)])
        mg = moral_graph(adj)
        # Fork: no co-parents, just undirected
        assert mg[0, 1] == 1
        assert mg[0, 2] == 1
        assert mg[1, 2] == 0

    def test_diamond_moral(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        mg = moral_graph(adj)
        # Co-parents of 3 are 1 and 2 → married
        assert mg[1, 2] == 1 and mg[2, 1] == 1
        # Symmetric
        assert np.array_equal(mg, mg.T)

    def test_moral_is_symmetric(self) -> None:
        from tests.conftest import random_dag
        adj = random_dag(8, seed=42)
        mg = moral_graph(adj)
        np.testing.assert_array_equal(mg, mg.T)

    def test_moral_no_self_loops(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        mg = moral_graph(adj)
        for i in range(4):
            assert mg[i, i] == 0

    def test_empty_dag_moral(self) -> None:
        adj = _adj(3, [])
        mg = moral_graph(adj)
        assert mg.sum() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Treewidth computation on known graphs
# ═══════════════════════════════════════════════════════════════════════════


class TestTreewidth:
    """Treewidth on graphs with known treewidth."""

    def test_tree_treewidth_is_1(self) -> None:
        # Tree: 0-1, 1-2, 1-3
        tree_adj = _undirected(4, [(0, 1), (1, 2), (1, 3)])
        tw = treewidth(tree_adj)
        assert tw == 1

    def test_path_treewidth_is_1(self) -> None:
        path = _undirected(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        tw = treewidth(path)
        assert tw == 1

    def test_cycle_treewidth_is_2(self) -> None:
        cyc = _undirected(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        tw = treewidth(cyc)
        assert tw == 2

    def test_complete_graph_treewidth(self) -> None:
        # K_n has treewidth n-1
        n = 4
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        K4 = _undirected(n, edges)
        tw = treewidth(K4)
        assert tw == n - 1

    def test_single_node_treewidth(self) -> None:
        g = np.zeros((1, 1), dtype=np.int8)
        tw = treewidth(g)
        assert tw <= 1  # either 0 or 1

    def test_empty_graph_treewidth(self) -> None:
        g = np.zeros((4, 4), dtype=np.int8)
        tw = treewidth(g)
        assert tw <= 1  # isolated nodes

    def test_grid_2x3_treewidth(self) -> None:
        # 2x3 grid: treewidth = 2
        # Nodes: 0 1 2
        #        3 4 5
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        grid = _undirected(6, edges)
        tw = treewidth(grid)
        assert tw <= 3  # exact is 2, heuristic may give upper bound

    def test_treewidth_of_dag(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        tw = treewidth_of_dag(adj)
        assert tw >= 1

    def test_treewidth_feasibility(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        feasible, tw = treewidth_feasibility(adj, max_treewidth=10)
        assert isinstance(feasible, bool)
        assert isinstance(tw, int)


# ═══════════════════════════════════════════════════════════════════════════
# Tree decomposition validity
# ═══════════════════════════════════════════════════════════════════════════


class TestTreeDecomposition:
    """Validate tree decomposition properties."""

    def _validate_td(self, td: TreeDecomposition, adj: np.ndarray) -> None:
        n = adj.shape[0]
        # 1. Every node appears in at least one bag
        all_nodes = set()
        for bag in td.bags:
            all_nodes.update(bag)
        for v in range(n):
            assert v in all_nodes, f"Node {v} missing from tree decomposition"

        # 2. Every edge is covered by some bag
        for i in range(n):
            for j in range(n):
                if adj[i, j] and i != j:
                    covered = any(i in bag and j in bag for bag in td.bags)
                    assert covered, f"Edge ({i},{j}) not covered"

        # 3. Running intersection property
        for v in range(n):
            containing_bags = [i for i, bag in enumerate(td.bags) if v in bag]
            if len(containing_bags) <= 1:
                continue
            # The subgraph induced by bags containing v should be connected
            visited = {containing_bags[0]}
            stack = [containing_bags[0]]
            while stack:
                b = stack.pop()
                for nb in td.tree_adj[b]:
                    if nb in set(containing_bags) and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            assert visited == set(containing_bags), (
                f"Running intersection violated for node {v}"
            )

    def test_chain_td(self) -> None:
        mg = _undirected(4, [(0, 1), (1, 2), (2, 3)])
        td = tree_decomposition(mg)
        self._validate_td(td, mg)
        assert td.width == 1

    def test_diamond_moral_td(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        mg = moral_graph(adj)
        td = tree_decomposition(mg)
        self._validate_td(td, mg)

    def test_complete_td(self) -> None:
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        K4 = _undirected(4, edges)
        td = tree_decomposition(K4)
        self._validate_td(td, K4)
        assert td.width == 3

    def test_td_width_attribute(self) -> None:
        mg = _undirected(4, [(0, 1), (1, 2), (2, 3)])
        td = tree_decomposition(mg)
        max_bag_size = max(len(bag) for bag in td.bags)
        assert td.width == max_bag_size - 1


# ═══════════════════════════════════════════════════════════════════════════
# Chordal and PEO
# ═══════════════════════════════════════════════════════════════════════════


class TestChordal:
    def test_tree_is_chordal(self) -> None:
        tree = _undirected(4, [(0, 1), (1, 2), (1, 3)])
        assert is_chordal(tree)

    def test_cycle4_not_chordal(self) -> None:
        cyc = _undirected(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        assert not is_chordal(cyc)

    def test_complete_is_chordal(self) -> None:
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        K4 = _undirected(4, edges)
        assert is_chordal(K4)

    def test_peo_exists_for_chordal(self) -> None:
        tree = _undirected(4, [(0, 1), (1, 2), (1, 3)])
        peo = perfect_elimination_ordering(tree)
        assert peo is not None
        assert len(peo) == 4

    def test_peo_none_for_non_chordal(self) -> None:
        cyc = _undirected(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        peo = perfect_elimination_ordering(cyc)
        assert peo is None

    def test_chordal_completion(self) -> None:
        cyc = _undirected(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        completed = chordal_completion(cyc)
        assert is_chordal(completed)
        # Has at least as many edges as original
        assert completed.sum() >= cyc.sum()


# ═══════════════════════════════════════════════════════════════════════════
# Maximal cliques
# ═══════════════════════════════════════════════════════════════════════════


class TestMaximalCliques:
    def test_triangle_clique(self) -> None:
        tri = _undirected(3, [(0, 1), (1, 2), (0, 2)])
        cliques = maximal_cliques(tri)
        assert len(cliques) >= 1
        assert any(len(c) == 3 for c in cliques)

    def test_path_cliques(self) -> None:
        path = _undirected(3, [(0, 1), (1, 2)])
        cliques = maximal_cliques(path)
        # Two maximal cliques: {0,1} and {1,2}
        assert len(cliques) == 2

    def test_empty_graph_cliques(self) -> None:
        g = np.zeros((3, 3), dtype=np.int8)
        cliques = maximal_cliques(g)
        assert len(cliques) == 3  # each node is its own clique


# ═══════════════════════════════════════════════════════════════════════════
# Junction tree
# ═══════════════════════════════════════════════════════════════════════════


class TestJunctionTree:
    def test_junction_tree_basic(self) -> None:
        # Chordalize a small graph first
        g = _undirected(4, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
        jt = junction_tree(g)
        assert isinstance(jt, JunctionTree)
        assert len(jt.cliques) >= 1

    def test_junction_tree_separators(self) -> None:
        g = _undirected(4, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)])
        jt = junction_tree(g)
        for (i, j), sep in jt.separators.items():
            # Separator is subset of both cliques
            assert sep <= jt.cliques[i]
            assert sep <= jt.cliques[j]

    def test_junction_tree_on_tree(self) -> None:
        tree = _undirected(4, [(0, 1), (1, 2), (1, 3)])
        jt = junction_tree(tree)
        assert len(jt.cliques) >= 1
